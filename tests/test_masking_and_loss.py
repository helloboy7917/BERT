import unittest

import torch

from config import BERTConfig
from data.masking import mask_tokens
from training.trainer import BERTTrainer


class TestMaskingAndLoss(unittest.TestCase):
    def test_random_replacements_include_regular_ids_below_special_ids(self):
        config = BERTConfig(
            vocab_size=10,
            pad_token_id=9,
            cls_token_id=8,
            sep_token_id=7,
            mask_token_id=6,
            unk_token_id=5,
        )
        input_ids = torch.zeros(1000, dtype=torch.long)
        masked, labels = mask_tokens(
            input_ids, config, mlm_probability=1.0,
            generator=torch.Generator().manual_seed(42),
        )

        self.assertTrue(torch.equal(labels, input_ids))
        self.assertTrue(torch.equal(input_ids, torch.zeros_like(input_ids)))
        self.assertTrue(((masked <= 4) | (masked == 6)).all().item())
        self.assertTrue(((masked >= 1) & (masked <= 4)).any().item())

    def test_all_special_tokens_allow_nsp_training(self):
        config = BERTConfig(vocab_size=10)
        input_ids = torch.tensor([[config.cls_token_id, config.sep_token_id]])
        _, labels = mask_tokens(input_ids, config)
        self.assertTrue((labels == -100).all().item())

        scores = torch.randn(1, 2, 10, requires_grad=True)
        nsp_logits = torch.randn(1, 2, requires_grad=True)
        outputs = {
            "prediction_scores": scores,
            "seq_relationship_score": nsp_logits,
        }
        batch = {"mlm_labels": labels, "nsp_label": torch.tensor([0])}
        trainer = BERTTrainer.__new__(BERTTrainer)
        total, mlm, nsp = trainer.compute_loss(outputs, batch)

        self.assertEqual(mlm.item(), 0.0)
        self.assertTrue(torch.isfinite(total).item())
        self.assertEqual(total.item(), nsp.item())
        total.backward()
        self.assertIsNotNone(nsp_logits.grad)
        self.assertTrue(torch.isfinite(nsp_logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()
