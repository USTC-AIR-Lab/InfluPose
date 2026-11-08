"""One 50-epoch run: 30 base epochs followed by 20 refinement epochs."""

import torch

from shared.ema import ExponentialMovingAverage
from stage2.auxiliary_losses import detached_state_dict, frozen_reference


def make_optimizer(model, cfg, *, refining=False):
    heads = [parameter for name in ("canonical_head", "inlier_head", "precision_head", "information_head", "size_head")
             for parameter in getattr(model, name).parameters()]
    head_ids = {id(parameter) for parameter in heads}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    learning_rate = cfg.refine_learning_rate if refining else cfg.learning_rate
    head_rate = cfg.refine_head_learning_rate if refining else cfg.head_learning_rate
    return torch.optim.AdamW([
        {"name": "point_information_backbone", "params": backbone, "lr": learning_rate},
        {"name": "correspondence_uncertainty_information_heads", "params": heads, "lr": head_rate},
    ], weight_decay=cfg.weight_decay)


def make_scheduler(optimizer, cfg, *, refining=False):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.base_epochs if refining else cfg.base_epochs,
        eta_min=cfg.refine_eta_min if refining else 0.1 * cfg.learning_rate,
    )


def begin_refinement(model, ema, cfg):
    """Freeze epoch-30 EMA and start the original low-LR refinement phase."""
    ema.copy_to(model.parameters())
    reference = frozen_reference(model)
    reference_state = detached_state_dict(reference)
    optimizer = make_optimizer(model, cfg, refining=True)
    scheduler = make_scheduler(optimizer, cfg, refining=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema_decay)
    return optimizer, scheduler, ema, reference, reference_state
