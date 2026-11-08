"""Stage2 execution layouts with a fixed global batch of 36.

Six processes with batch six reproduce the reference layout. A single process
with batch 36 preserves the batch size but changes the numerical trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

EXECUTION_PROTOCOL_FORMAT = "STAGE2_PAPER_EXECUTION_PROTOCOL_V1"
FRESH_SINGLE_GPU_LAYOUT = "fresh_single_gpu_batch36_global36"
LEGACY_DDP_LAYOUT = "legacy_ddp_6x6_global36"
GLOBAL_BATCH_SIZE = 36
MAXIMUM_FRESH_PARALLEL_CHECKPOINT_JOBS = 6


@dataclass(frozen=True)
class ExecutionLayout:
    """One of the two protocol-qualified Stage2 execution layouts."""

    name: str
    world_size: int
    per_device_batch_size: int
    global_batch_size: int
    distributed: bool

    def metadata(self, *, origin: str) -> dict[str, object]:
        return {
            "format": EXECUTION_PROTOCOL_FORMAT,
            "execution_layout": self.name,
            "resolved_world_size": self.world_size,
            "per_device_batch_size": self.per_device_batch_size,
            "resolved_global_batch_size": self.global_batch_size,
            "distributed": self.distributed,
            "metadata_origin": origin,
            "bit_exact_equivalence_to_other_layout_claimed": False,
        }


FRESH_SINGLE_GPU = ExecutionLayout(
    name=FRESH_SINGLE_GPU_LAYOUT,
    world_size=1,
    per_device_batch_size=36,
    global_batch_size=GLOBAL_BATCH_SIZE,
    distributed=False,
)
LEGACY_DDP = ExecutionLayout(
    name=LEGACY_DDP_LAYOUT,
    world_size=6,
    per_device_batch_size=6,
    global_batch_size=GLOBAL_BATCH_SIZE,
    distributed=True,
)
LAYOUTS = {
    FRESH_SINGLE_GPU.name: FRESH_SINGLE_GPU,
    LEGACY_DDP.name: LEGACY_DDP,
}


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} is not an integer: {value!r}") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def resolve_execution_layout(
    *,
    world_size: int,
    per_device_batch_size: int,
    distributed: bool,
) -> ExecutionLayout:
    """Resolve an exact legal layout and reject all intermediate mixtures."""

    resolved_world_size = _positive_int(world_size, name="world_size")
    resolved_batch = _positive_int(
        per_device_batch_size, name="per_device_batch_size"
    )
    resolved_distributed = bool(distributed)
    signature = (resolved_world_size, resolved_batch, resolved_distributed)
    for layout in LAYOUTS.values():
        if signature == (
            layout.world_size,
            layout.per_device_batch_size,
            layout.distributed,
        ):
            return layout
    raise ValueError(
        "unsupported Stage2 execution layout: expected fresh single-GPU "
        "world_size=1/batch=36 or explicit legacy DDP "
        "world_size=6/batch=6, both at global batch 36; got "
        f"world_size={resolved_world_size}, batch={resolved_batch}, "
        f"distributed={resolved_distributed}"
    )


def record_runtime_execution_layout(
    cfg: object, *, world_size: int, distributed: bool
) -> dict[str, object]:
    """Resolve and attach checkpoint-serializable runtime metadata to ``cfg``."""

    layout = resolve_execution_layout(
        world_size=world_size,
        per_device_batch_size=cfg.batch_size,
        distributed=distributed,
    )
    # ``vars(cfg)`` is serialized into training_config by the Stage2 trainer.
    cfg.distributed = layout.distributed
    cfg.execution_protocol_format = EXECUTION_PROTOCOL_FORMAT
    cfg.execution_layout = layout.name
    cfg.resolved_world_size = layout.world_size
    cfg.resolved_global_batch_size = layout.global_batch_size
    return layout.metadata(origin="runtime_resolved_and_checkpoint_recorded")


def _mapping(state: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint lacks mapping {key!r}")
    return value


def checkpoint_execution_layout(
    state: Mapping[str, object],
    *,
    allow_legacy_missing_metadata: bool = False,
    required_layout: str | None = None,
) -> dict[str, object]:
    """Validate execution metadata recorded in a Stage2 checkpoint.

    Earlier checkpoints may omit these fields. They may be interpreted as
    6x6 DDP only when the caller opts in with
    ``allow_legacy_missing_metadata=True``. Missing fields are never inferred
    for a new checkpoint.
    """

    training = _mapping(state, "training_config")
    provenance = _mapping(state, "provenance")
    fields = (
        "execution_protocol_format",
        "execution_layout",
        "resolved_world_size",
        "resolved_global_batch_size",
    )
    present = {
        (container_name, field)
        for container_name, container in (
            ("training_config", training),
            ("provenance", provenance),
        )
        for field in fields
        if field in container
    }
    if not present:
        if not allow_legacy_missing_metadata:
            raise ValueError(
                "checkpoint lacks explicit execution-layout metadata; legacy "
                "inference is forbidden for fresh current-queue artifacts"
            )
        if training.get("distributed") is not True:
            raise ValueError(
                "legacy execution compatibility requires distributed=true"
            )
        if _positive_int(training.get("batch_size"), name="batch_size") != 6:
            raise ValueError("legacy execution compatibility requires batch_size=6")
        layout = LEGACY_DDP
        metadata = layout.metadata(
            origin="authorized_historical_inference_from_archived_6way_ddp_contract"
        )
        metadata["legacy_checkpoint_missing_explicit_fields"] = True
    else:
        expected_present = {
            (container_name, field)
            for container_name in ("training_config", "provenance")
            for field in fields
        }
        missing = sorted(expected_present - present)
        if missing:
            rendered = ", ".join(f"{container}.{field}" for container, field in missing)
            raise ValueError(f"checkpoint has partial execution metadata; missing {rendered}")
        for field in fields:
            if training.get(field) != provenance.get(field):
                raise ValueError(
                    f"training_config/provenance execution field {field!r} disagrees"
                )
        if training.get("execution_protocol_format") != EXECUTION_PROTOCOL_FORMAT:
            raise ValueError("checkpoint execution protocol format mismatch")
        layout_name = str(training.get("execution_layout"))
        if layout_name not in LAYOUTS:
            raise ValueError(f"unknown checkpoint execution layout {layout_name!r}")
        distributed = training.get("distributed")
        if not isinstance(distributed, bool):
            raise TypeError("checkpoint training_config.distributed is not boolean")
        layout = resolve_execution_layout(
            world_size=_positive_int(
                training.get("resolved_world_size"), name="resolved_world_size"
            ),
            per_device_batch_size=_positive_int(
                training.get("batch_size"), name="batch_size"
            ),
            distributed=distributed,
        )
        if layout.name != layout_name:
            raise ValueError("checkpoint execution-layout label disagrees with its values")
        recorded_global_batch = _positive_int(
            training.get("resolved_global_batch_size"),
            name="resolved_global_batch_size",
        )
        if recorded_global_batch != layout.global_batch_size:
            raise ValueError("checkpoint resolved global batch is inconsistent")
        metadata = layout.metadata(origin="explicit_checkpoint_metadata")
        metadata["legacy_checkpoint_missing_explicit_fields"] = False

    if required_layout is not None and metadata["execution_layout"] != required_layout:
        raise ValueError(
            "checkpoint execution layout mismatch: "
            f"{metadata['execution_layout']!r} != {required_layout!r}"
        )
    return metadata


def aggregate_execution_protocol(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return one aggregate layout, failing closed on fresh/legacy mixing."""

    if not records:
        raise ValueError("cannot aggregate an empty execution protocol")
    keys = (
        "execution_layout",
        "resolved_world_size",
        "per_device_batch_size",
        "resolved_global_batch_size",
        "distributed",
    )
    signatures = {tuple(record.get(key) for key in keys) for record in records}
    if len(signatures) != 1:
        raise ValueError(
            "fresh single-GPU and legacy DDP execution layouts cannot be mixed "
            "in one fixed-configuration protocol summary"
        )
    signature = next(iter(signatures))
    layout_name = str(signature[0])
    if layout_name not in LAYOUTS:
        raise ValueError(f"aggregate uses unknown execution layout {layout_name!r}")
    layout = LAYOUTS[layout_name]
    expected = (
        layout.name,
        layout.world_size,
        layout.per_device_batch_size,
        layout.global_batch_size,
        layout.distributed,
    )
    if signature != expected:
        raise ValueError("aggregate execution values disagree with its layout label")
    fresh = layout.name == FRESH_SINGLE_GPU_LAYOUT
    return {
        "format": EXECUTION_PROTOCOL_FORMAT,
        "execution_layout": layout.name,
        "resolved_world_size": layout.world_size,
        "per_device_batch_size": layout.per_device_batch_size,
        "resolved_global_batch_size": layout.global_batch_size,
        "distributed": layout.distributed,
        "execution_layouts_mixed": False,
        "maximum_parallel_independent_checkpoint_jobs": (
            MAXIMUM_FRESH_PARALLEL_CHECKPOINT_JOBS if fresh else 1
        ),
        "one_gpu_per_independent_checkpoint_job": fresh,
        "global_batch_held_constant_at_36": True,
        "epochs_optimizer_steps_and_learning_rates_held_constant": True,
        "bit_exact_equivalence_to_other_layout_claimed": False,
        "numerical_trajectory_note": (
            "single-GPU and six-way DDP runs share the scientific global-batch, "
            "epoch, optimizer-step and learning-rate contract, but their random "
            "trajectories and floating-point reductions are not guaranteed bit-exact"
        ),
    }
