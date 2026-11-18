
import torch


def build_optimizer(
    model,
    learning_rate=5e-5,
    weight_decay=0.01,
    adam_epsilon=1e-8
):

    no_decay = [
        "bias",
        "layernorm.weight",
        "layer_norm.weight"
    ]

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        parameter_name = name.lower()

        if any(
            key in parameter_name
            for key in no_decay
        ):
            no_decay_parameters.append(parameter)

        else:
            decay_parameters.append(parameter)

    optimizer_grouped_parameters = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0
        }
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_epsilon
    )

    return optimizer
