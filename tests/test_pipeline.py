import json
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader

from config import BERTConfig
from data.dataset import BERTPretrainingDataset, encode_sequence
from data.prepare_corpus import prepare_corpus
from train import ResumableBatchSampler, split_documents
from training.trainer import BERTTrainer


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = 15
        self.pad_token_id, self.cls_token_id, self.sep_token_id = 0, 1, 2
        self.mask_token_id, self.unk_token_id = 3, 4

    def encode(self, text, add_special_tokens=False):
        ids = [3 if word == "[MASK]" else 5 + sum(map(ord, word)) % 10
               for word in text.split()]
        return FakeEncoding(ids)

    def id_to_token(self, index):
        return str(index)


DOCUMENTS = [{"sentences": [f"document {i} first", f"document {i} second",
                             f"document {i} third"]} for i in range(6)]


class TestPipelineData(unittest.TestCase):
    def test_preparation_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, docs, corpus = (root / name for name in ("raw.jsonl", "docs.jsonl", "corpus.txt"))
            source.write_text(json.dumps({"sentences": [" one ", "", "two"]}) + "\n", encoding="utf-8")
            self.assertEqual(prepare_corpus(source, docs, corpus), 1)
            self.assertEqual(json.loads(docs.read_text(encoding="utf-8"))["sentences"], ["one", "two"])
            self.assertEqual(corpus.read_text(encoding="utf-8"), "one\ntwo\n")
            source.write_text('{"sentences": [1]}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ":1:"):
                prepare_corpus(source, docs, corpus)
            self.assertEqual(corpus.read_text(encoding="utf-8"), "one\ntwo\n")
            with self.assertRaises(ValueError):
                prepare_corpus(source, source, corpus)

    def test_split_and_dataset(self):
        train, validation = split_documents(DOCUMENTS, 0.1, 42)
        self.assertGreaterEqual(len(train), 2)
        self.assertGreaterEqual(len(validation), 2)
        self.assertFalse({d["sentences"][0] for d in train} & {d["sentences"][0] for d in validation})
        self.assertEqual((train, validation), split_documents(DOCUMENTS, 0.1, 42))
        with self.assertRaises(ValueError):
            split_documents(DOCUMENTS[:3], 0.1, 42)
        config = BERTConfig(vocab_size=15, max_position_embeddings=8)
        dataset = BERTPretrainingDataset(train, FakeTokenizer(), config, 8)
        item = dataset[0]
        self.assertTrue(all(torch.equal(item[k], dataset[0][k]) for k in item))
        self.assertEqual(item["input_ids"].shape, (8,))
        self.assertIn(item["nsp_label"].item(), (0, 1))
        self.assertTrue((item["mlm_labels"][item["attention_mask"] == 0] == -100).all())
        pair = encode_sequence(FakeTokenizer(), config, "a b c d", "e f g h", 7)
        self.assertEqual(pair["input_ids"].count(config.sep_token_id), 2)
        self.assertEqual(pair["token_type_ids"][-1], 1)
        with self.assertRaises(ValueError):
            BERTPretrainingDataset(train[:1], FakeTokenizer(), config, 8)

    def test_sampler_resume(self):
        sampler = ResumableBatchSampler(list(range(11)), 3, 42)
        sampler.set_epoch(1)
        full = list(sampler)
        sampler.set_epoch(1, 2)
        self.assertEqual(list(sampler), full[2:])

    def test_nsp_examples_follow_document_boundaries(self):
        documents = [{"sentences": ["aaaa", "bbbb"]}, {"sentences": ["cccc", "dddd"]}]
        config = BERTConfig(vocab_size=15, max_position_embeddings=8)
        seen = set()
        with mock.patch("data.dataset.mask_tokens", side_effect=lambda ids, *a, **kw: (ids, ids)):
            for seed in range(30):
                dataset = BERTPretrainingDataset(documents, FakeTokenizer(), config, 8, seed=seed)
                for index in range(2):
                    item = dataset[index]
                    ids = item["input_ids"].tolist()
                    second = ids[ids.index(config.sep_token_id) + 1]
                    label = item["nsp_label"].item()
                    positive = FakeTokenizer().encode(documents[index]["sentences"][1]).ids[0]
                    negatives = {FakeTokenizer().encode(sentence).ids[0]
                                 for sentence in documents[1 - index]["sentences"]}
                    self.assertEqual(second == positive, label == 0)
                    if label == 1:
                        self.assertIn(second, negatives)
                    seen.add(label)
        self.assertEqual(seen, {0, 1})

    def test_worker_count_does_not_change_examples(self):
        dataset = BERTPretrainingDataset(DOCUMENTS, FakeTokenizer(),
                                         BERTConfig(vocab_size=15, max_position_embeddings=8), 8)
        dataset.set_epoch(2)
        single = list(DataLoader(dataset, batch_size=2, num_workers=0))
        workers = list(DataLoader(dataset, batch_size=2, num_workers=2))
        for first, second in zip(single, workers):
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]))


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BERTConfig(vocab_size=15, max_position_embeddings=8)
        self.weight = torch.nn.Parameter(torch.tensor(0.4))
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch, length = input_ids.shape
        value = self.dropout(torch.ones(batch, length)) * self.weight
        scores = value.unsqueeze(-1) * torch.arange(15, dtype=torch.float)
        logits = torch.stack((value[:, 0], -value[:, 0]), dim=-1)
        return {"prediction_scores": scores, "seq_relationship_score": logits}


class TestResume(unittest.TestCase):
    def make(self, folder):
        dataset = BERTPretrainingDataset(DOCUMENTS, FakeTokenizer(), BERTConfig(vocab_size=15, max_position_embeddings=8), 8)
        sampler = ResumableBatchSampler(dataset, 2, 7)
        loader = DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(9))
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        trainer = BERTTrainer(model, optimizer, scheduler, "cpu", output_dir=folder,
                              gradient_accumulation_steps=2, resume_metadata={"test": 1})
        return trainer, loader

    def test_mid_epoch_checkpoint_restores_toy_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.manual_seed(123)
            full, full_loader = self.make(root / "full")
            full_history = full.fit(full_loader, 1)
            torch.manual_seed(123)
            interrupted, loader = self.make(root / "interrupted")
            original_save = interrupted.save_checkpoint
            def save_then_interrupt(*args, **kwargs):
                original_save(*args, **kwargs)
                raise RuntimeError("simulated interruption")
            interrupted.save_checkpoint = save_then_interrupt
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                interrupted.fit(loader, 1)
            resumed, loader = self.make(root / "resumed")
            state = resumed.load_checkpoint(root / "interrupted" / "last.pt")
            self.assertEqual(state["next_batch"], 2)
            history = resumed.fit(loader, 1, resume_state=state)
            self.assertEqual(full_history, history)
            self.assertTrue(torch.equal(full.model.weight, resumed.model.weight))
            self.assertEqual(full.scheduler.state_dict(), resumed.scheduler.state_dict())
            resumed.resume_metadata = {"test": 2}
            with self.assertRaisesRegex(ValueError, "do not match"):
                resumed.load_checkpoint(root / "interrupted" / "last.pt")

    def test_evaluate_leaves_weights_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer, loader = self.make(folder)
            before = trainer.model.weight.detach().clone()
            metrics = trainer.evaluate(loader)
            self.assertEqual(set(metrics), {"total_loss", "mlm_loss", "nsp_loss"})
            self.assertFalse(trainer.model.training)
            self.assertTrue(torch.equal(before, trainer.model.weight))


class TestInference(unittest.TestCase):
    def test_legacy_checkpoint_and_input_validation(self):
        # The optional tokenizers package is replaced here by an in-memory stub;
        # no tokenizer is trained and no external files are downloaded.
        from dataclasses import asdict
        from models.bert import BERTForPreTraining
        fake_module = types.ModuleType("tokenizers")
        fake_module.Tokenizer = object
        with mock.patch.dict(sys.modules, {"tokenizers": fake_module}):
            inference_module = importlib.import_module("inference")
            try:
                with tempfile.TemporaryDirectory() as folder, mock.patch.object(
                        inference_module, "BertTokenizer", side_effect=lambda _: FakeTokenizer()):
                    config = BERTConfig(vocab_size=15, hidden_size=8, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=16, max_position_embeddings=8)
                    path = Path(folder) / "model.pt"
                    torch.save({"config": asdict(config),
                                "model_state_dict": BERTForPreTraining(config).state_dict()}, path)
                    inference = inference_module.BERTInference(path, "unused", "cpu")
                    self.assertEqual(len(inference.predict_masked_tokens("hello [MASK]", top_k=3)[0]["predictions"]), 3)
                    self.assertIn(inference.predict_next_sentence("a b c d", "e f g h")["label"], ("IsNext", "NotNext"))
                    pair = inference.encode("a b c d", "e f g h")
                    self.assertEqual((pair["input_ids"] == config.sep_token_id).sum().item(), 2)
                    with self.assertRaises(ValueError):
                        inference.predict_masked_tokens("hello", top_k=1)
                    with self.assertRaises(ValueError):
                        inference.predict_masked_tokens("hello [MASK]", top_k=0)
                    config.mask_token_id = 14
                    torch.save({"config": asdict(config), "model_state_dict": {}}, path)
                    with self.assertRaisesRegex(ValueError, "mask_token_id"):
                        inference_module.BERTInference(path, "unused", "cpu")
            finally:
                sys.modules.pop("inference", None)
                sys.modules.pop("tokenizer.tokenizer", None)


class TestRealTokenizer(unittest.TestCase):
    def test_local_wordpiece_file_without_training(self):
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, processors
        except ImportError:
            self.skipTest("tokenizers is not installed")
        from dataclasses import asdict
        from models.bert import BERTForPreTraining
        from tokenizer.tokenizer import BertTokenizer
        from inference import BERTInference
        vocab = {token: i for i, token in enumerate(
            ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "hello", "world", "next"])}
        tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
        with tempfile.TemporaryDirectory() as folder:
            tokenizer_path = Path(folder) / "tokenizer.json"
            tokenizer.save(str(tokenizer_path))
            wrapper = BertTokenizer(tokenizer_path)
            config = BERTConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=16, max_position_embeddings=8)
            dataset = BERTPretrainingDataset(
                [{"sentences": ["hello world", "next world"]},
                 {"sentences": ["world hello", "hello next"]}], wrapper, config, 8)
            self.assertEqual(dataset[0]["input_ids"].shape, (8,))
            path = Path(folder) / "model.pt"
            torch.save({"config": asdict(config),
                        "model_state_dict": BERTForPreTraining(config).state_dict()}, path)
            inference = BERTInference(path, tokenizer_path, "cpu")
            self.assertEqual(len(inference.predict_masked_tokens("hello [MASK]", 2)[0]["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()
