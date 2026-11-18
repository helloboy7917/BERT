
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps
):

    if num_training_steps <= 0:
        raise ValueError(
            "num_training_steps must be greater than 0"
        )

    if not 0 <= num_warmup_steps <= num_training_steps:
        raise ValueError(
            "Invalid num_warmup_steps"
        )

    def lr_lambda(current_step):

        # Warmup
        if current_step < num_warmup_steps:

            return float(current_step) / max(
                1,
                num_warmup_steps
            )

        # Linear Decay
        progress = (
            current_step - num_warmup_steps
        ) / max(
            1,
            num_training_steps - num_warmup_steps
        )

        return max(
            0.0,
            1.0 - progress
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda
    )

    return scheduler
