
import torch

from config import BERTConfig


def mask_tokens(
    input_ids,
    config: BERTConfig,
    mlm_probability=0.15,
    generator=None
):
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must be torch.long")

    if input_ids.dim() not in (1, 2):
        raise ValueError("input_ids must be 1D or 2D")

    if not 0 < mlm_probability <= 1:
        raise ValueError("Invalid MLM probability")

    # 1. 입력 형태 통일
    was_1d = input_ids.dim() == 1

    if was_1d:
        input_ids = input_ids.unsqueeze(0)

    masked_input_ids = input_ids.clone()

    # 2. 정답 Tensor 생성
    mlm_labels = torch.full_like(
        input_ids,
        fill_value=-100
    )

    # 3. Special Tokens
    special_ids = [
        config.pad_token_id,
        config.cls_token_id,
        config.sep_token_id,
        config.mask_token_id,
        config.unk_token_id
    ]

    regular_ids = torch.tensor(
        [token_id for token_id in range(config.vocab_size)
         if token_id not in special_ids],
        dtype=torch.long,
        device=input_ids.device
    )

    if regular_ids.numel() == 0:
        raise ValueError("No regular tokens in vocabulary")

    # 4. Batch별 Masking
    for batch_idx, tokens in enumerate(input_ids):

        # 특수 토큰 제외
        candidate_mask = torch.ones_like(
            tokens,
            dtype=torch.bool
        )

        for token_id in special_ids:
            candidate_mask &= tokens != token_id

        candidate_positions = torch.where(
            candidate_mask
        )[0]

        if candidate_positions.numel() == 0:
            continue

        # 5. 약 15% 토큰 선택
        num_to_mask = max(
            1,
            round(candidate_positions.numel() * mlm_probability)
        )

        permutation = torch.randperm(
            candidate_positions.numel(),
            device=tokens.device,
            generator=generator
        )

        selected_positions = candidate_positions[
            permutation[:num_to_mask]
        ]

        # 6. 원본 Token ID를 정답으로 저장
        mlm_labels[
            batch_idx,
            selected_positions
        ] = tokens[selected_positions]

        # 7. 80 / 10 / 10
        probs = torch.rand(
            num_to_mask,
            device=tokens.device,
            generator=generator
        )

        # 80% → [MASK]
        mask_positions = selected_positions[
            probs < 0.8
        ]

        masked_input_ids[
            batch_idx,
            mask_positions
        ] = config.mask_token_id

        # 10% → Random Token
        random_positions = selected_positions[
            (probs >= 0.8) & (probs < 0.9)
        ]

        if random_positions.numel() > 0:

            random_indices = torch.randint(
                low=0,
                high=regular_ids.numel(),
                size=(random_positions.numel(),),
                device=tokens.device,
                generator=generator
            )

            masked_input_ids[
                batch_idx,
                random_positions
            ] = regular_ids[random_indices]

        # 나머지 10% → 원본 유지

    # 8. 원래 입력 차원으로 반환
    if was_1d:
        return (
            masked_input_ids.squeeze(0),
            mlm_labels.squeeze(0)
        )

    return masked_input_ids, mlm_labels
