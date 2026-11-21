"""BERT pretraining examples with reproducible NSP and dynamic MLM."""

import random

import torch
from torch.utils.data import Dataset

from data.masking import mask_tokens


def encode_sequence(tokenizer, config, text, text_pair=None, max_length=None):
    """Build a BERT input, truncating content while preserving separators."""
    limit = max_length if max_length is not None else config.max_position_embeddings
    minimum = 3 if text_pair is not None else 2
    if limit < minimum or limit > config.max_position_embeddings:
        raise ValueError(f"max_length must be between {minimum} and max_position_embeddings")
    a = list(tokenizer.tokenizer.encode(text, add_special_tokens=False).ids)
    b = (list(tokenizer.tokenizer.encode(text_pair, add_special_tokens=False).ids)
         if text_pair is not None else None)
    while len(a) + (len(b) if b is not None else 0) + minimum > limit:
        if b is not None and len(b) > len(a):
            b.pop()
        else:
            a.pop()
    ids = [config.cls_token_id] + a + [config.sep_token_id]
    types = [0] * len(ids)
    if b is not None:
        ids += b + [config.sep_token_id]
        types += [1] * (len(b) + 1)
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": [1] * len(ids)}


class BERTPretrainingDataset(Dataset):
    def __init__(self, documents, tokenizer, config, max_length, seed=42, epoch=0):
        if not 3 <= max_length <= config.max_position_embeddings:
            raise ValueError("max_length must be between 3 and max_position_embeddings")
        self.documents = []
        for document in documents:
            if not isinstance(document, dict) or not isinstance(document.get("sentences"), list):
                raise ValueError("Each document must have a sentences array")
            if any(not isinstance(s, str) for s in document["sentences"]):
                raise ValueError("Sentences must be strings")
            sentences = [s.strip() for s in document["sentences"] if s.strip()]
            if sentences:
                self.documents.append(sentences)
        if len(self.documents) < 2:
            raise ValueError("NSP requires at least two non-empty documents per split")
        self.examples = [(doc_id, i) for doc_id, doc in enumerate(self.documents)
                         for i in range(len(doc) - 1)]
        if not self.examples:
            raise ValueError("Each split needs a document with two consecutive sentences")
        self.tokenizer, self.config, self.max_length = tokenizer, config, max_length
        self.seed, self.epoch = seed, epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # A stable per-example seed also avoids worker/prefetch RNG differences.
        sample_seed = (self.seed + self.epoch * 1_000_003 + index * 97_409) % (2**63 - 1)
        rng = random.Random(sample_seed)
        doc_id, sentence_id = self.examples[index]
        sentence_a = self.documents[doc_id][sentence_id]
        if rng.random() < 0.5:
            sentence_b = self.documents[doc_id][sentence_id + 1]
            nsp_label = 0
        else:
            other = rng.randrange(len(self.documents) - 1)
            if other >= doc_id:
                other += 1
            sentence_b = rng.choice(self.documents[other])
            nsp_label = 1
        encoded = encode_sequence(self.tokenizer, self.config, sentence_a, sentence_b, self.max_length)
        padding = self.max_length - len(encoded["input_ids"])
        encoded["input_ids"] += [self.config.pad_token_id] * padding
        encoded["token_type_ids"] += [0] * padding
        encoded["attention_mask"] += [0] * padding
        generator = torch.Generator().manual_seed(sample_seed)
        ids, labels = mask_tokens(torch.tensor(encoded["input_ids"], dtype=torch.long),
                                  self.config, generator=generator)
        return {"input_ids": ids, "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.long),
                "mlm_labels": labels, "nsp_label": torch.tensor(nsp_label, dtype=torch.long)}
