
import unittest

import torch

from config import BERTConfig
from models.bert import BERTForPreTraining


class TestBERTModel(unittest.TestCase):

    def test_forward(self):

        # 테스트용 작은 설정
        config = BERTConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=32
        )

        # CPU에서 테스트
        model = BERTForPreTraining(config)
        model.eval()

        # Batch Size = 2
        # Sequence Length = 8
        input_ids = torch.randint(
            low=5,
            high=config.vocab_size,
            size=(2, 8)
        )

        # 마지막 두 토큰은 Padding
        input_ids[:, -2:] = config.pad_token_id

        attention_mask = (
            input_ids != config.pad_token_id
        ).long()

        token_type_ids = torch.zeros_like(
            input_ids
        )

        # Forward Pass (학습 없음)
        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        # 출력 Shape 검증
        self.assertEqual(
            outputs["sequence_output"].shape,
            (2, 8, 64)
        )

        self.assertEqual(
            outputs["pooled_output"].shape,
            (2, 64)
        )

        self.assertEqual(
            outputs["prediction_scores"].shape,
            (2, 8, 128)
        )

        self.assertEqual(
            outputs["seq_relationship_score"].shape,
            (2, 2)
        )

        # Encoder Layer 개수
        self.assertEqual(
            len(outputs["attention_probs"]),
            2
        )

        # Attention Shape
        self.assertEqual(
            outputs["attention_probs"][0].shape,
            (2, 4, 8, 8)
        )

        # 결과값에 NaN이나 Inf가 없어야 함
        self.assertTrue(
            torch.isfinite(
                outputs["prediction_scores"]
            ).all().item()
        )

        # Padding 토큰에 Attention이 향하지 않는지 확인
        for probs in outputs["attention_probs"]:
            self.assertTrue(
                (probs[..., -2:] == 0).all().item()
            )

        # MLM Decoder와 Token Embedding 가중치 공유 확인
        self.assertIs(
            model.cls.predictions.decoder.weight,
            model.bert.embeddings.word_embeddings.weight
        )

        print("\nBERT Forward Test Passed!")
        print("Sequence:", outputs["sequence_output"].shape)
        print("Pooled:", outputs["pooled_output"].shape)
        print("MLM:", outputs["prediction_scores"].shape)
        print("NSP:", outputs["seq_relationship_score"].shape)


if __name__ == "__main__":
    unittest.main()
