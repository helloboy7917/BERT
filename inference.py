
import argparse

import torch

from config import BERTConfig
from tokenizer.tokenizer import BertTokenizer
from models.bert import BERTForPreTraining
from data.dataset import encode_sequence


class BERTInference:

    def __init__(
        self,
        checkpoint_path,
        tokenizer_path,
        device="auto"
    ):

        if device == "auto":
            device = (
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # Tokenizer
        self.tokenizer = BertTokenizer(
            tokenizer_path
        )

        # Checkpoint
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

        # Configuration
        self.config = BERTConfig(
            **checkpoint["config"]
        )

        if self.config.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "Tokenizer and model vocabulary sizes do not match"
            )

        for name in ("pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id", "unk_token_id"):
            if getattr(self.config, name) != getattr(self.tokenizer, name):
                raise ValueError(f"Tokenizer and model {name} do not match")

        # Model
        self.model = BERTForPreTraining(
            self.config
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

    def encode(self, text, text_pair=None):

        encoded = encode_sequence(self.tokenizer, self.config, text, text_pair)
        input_ids = encoded["input_ids"]
        token_type_ids = encoded["token_type_ids"]
        attention_mask = encoded["attention_mask"]

        return {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=self.device
            ),
            "token_type_ids": torch.tensor(
                [token_type_ids],
                dtype=torch.long,
                device=self.device
            ),
            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
                device=self.device
            )
        }

    @torch.no_grad()
    def predict_masked_tokens(
        self,
        text,
        top_k=5
    ):

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        inputs = self.encode(text)

        outputs = self.model(**inputs)

        input_ids = inputs["input_ids"][0]

        mask_positions = torch.where(
            input_ids == self.config.mask_token_id
        )[0]

        if mask_positions.numel() == 0:
            raise ValueError(
                "Input must contain [MASK] within the maximum sequence length"
            )

        results = []

        for position in mask_positions:

            logits = outputs[
                "prediction_scores"
            ][0, position]

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            values, indices = torch.topk(
                probabilities,
                k=min(top_k, self.config.vocab_size)
            )

            predictions = []

            for token_id, score in zip(
                indices.tolist(),
                values.tolist()
            ):

                token = self.tokenizer.tokenizer.id_to_token(
                    token_id
                )

                predictions.append({
                    "token": token,
                    "probability": score
                })

            results.append({
                "position": position.item(),
                "predictions": predictions
            })

        return results

    @torch.no_grad()
    def predict_next_sentence(
        self,
        sentence_a,
        sentence_b
    ):

        inputs = self.encode(
            sentence_a,
            sentence_b
        )

        outputs = self.model(**inputs)

        logits = outputs[
            "seq_relationship_score"
        ]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )[0]

        predicted_class = torch.argmax(
            probabilities
        ).item()

        label = (
            "IsNext"
            if predicted_class == 0
            else "NotNext"
        )

        return {
            "label": label,
            "is_next_probability": probabilities[0].item(),
            "not_next_probability": probabilities[1].item()
        }


def main():

    parser = argparse.ArgumentParser(
        description="BERT Inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/last.pt"
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto"
    )

    parser.add_argument(
        "--text",
        type=str
    )

    parser.add_argument(
        "--sentence-a",
        type=str
    )

    parser.add_argument(
        "--sentence-b",
        type=str
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    has_text = bool(args.text)
    has_pair = bool(args.sentence_a) and bool(args.sentence_b)
    if has_text == has_pair or (has_text and (args.sentence_a or args.sentence_b)):
        parser.error("Provide --text or both --sentence-a and --sentence-b")

    inference = BERTInference(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device
    )

    # MLM
    if args.text:

        results = inference.predict_masked_tokens(
            args.text,
            top_k=args.top_k
        )

        for result in results:

            print(f"\nPosition: {result['position']}")

            for prediction in result["predictions"]:

                print(
                    f"{prediction['token']}: "
                    f"{prediction['probability']:.4f}"
                )

    # NSP
    elif args.sentence_a and args.sentence_b:

        result = inference.predict_next_sentence(
            args.sentence_a,
            args.sentence_b
        )

        print("\nNSP Result")

        print(f"Prediction: {result['label']}")

        print(
            "IsNext Probability:",
            result["is_next_probability"]
        )

        print(
            "NotNext Probability:",
            result["not_next_probability"]
        )

    else:

        parser.error(
            "Provide --text or both --sentence-a and --sentence-b"
        )


if __name__ == "__main__":
    main()
