import math

import torch

from shared.point_features import estimate_camera_observation_features
from shared.pose_geometry import (
    axis_angle_to_matrix,
    canonicalize_continuous_y_rotation,
)
from stage2.model import (
    Stage2PoseInformationField,
    _skew,
    information_irls,
    weighted_kabsch,
)


def test_weighted_kabsch_recovers_a_rigid_transform():
    generator = torch.Generator().manual_seed(260925)
    source = torch.randn(2, 32, 3, generator=generator)
    rotation = torch.tensor(
        [
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ]
    )
    translation = torch.tensor([[0.1, -0.2, 0.8], [-0.3, 0.2, 1.1]])
    target = torch.einsum("bij,bkj->bki", rotation, source) + translation[:, None]
    solved_rotation, solved_translation, singular = weighted_kabsch(
        source, target, torch.ones(2, 32)
    )
    assert torch.allclose(solved_rotation, rotation, atol=1e-5)
    assert torch.allclose(solved_translation, translation, atol=1e-5)
    assert singular.shape == (2, 3)


def test_weighted_kabsch_reflection_branch_flips_the_smallest_direction():
    source = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )[None]
    improper = torch.diag(torch.tensor((1.0, 1.0, -1.0)))
    target = torch.einsum("ij,bkj->bki", improper, source)
    rotation, translation, singular = weighted_kabsch(
        source, target, torch.ones(1, source.shape[1])
    )

    assert torch.all(singular[:, :-1] > singular[:, 1:])
    assert torch.allclose(rotation, torch.eye(3)[None], atol=1e-6)
    assert torch.allclose(translation, torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(1), atol=1e-6)


def test_axis_angle_map_respects_the_small_angle_and_eight_degree_bounds():
    angles = torch.tensor(
        (
            0.0,
            5e-8,
            0.999999e-6,
            1e-6,
            1.000001e-6,
            0.10,
            math.radians(8.0),
        ),
        dtype=torch.float64,
    )
    increments = torch.zeros(angles.shape[0], 3, dtype=torch.float64)
    increments[:, 0] = angles
    matrices = axis_angle_to_matrix(increments)
    represented = torch.atan2(matrices[:, 2, 1], matrices[:, 1, 1]).abs()

    assert torch.isfinite(matrices).all()
    assert torch.all(represented <= angles + 1e-15)
    assert torch.allclose(represented[3:], angles[3:], atol=1e-15, rtol=1e-12)


def test_continuous_y_gauge_preserves_the_rotation_quotient_class():
    generator = torch.Generator().manual_seed(260934)
    axis_angle = torch.randn(64, 3, generator=generator, dtype=torch.float64)
    axis_angle = axis_angle / torch.linalg.vector_norm(
        axis_angle, dim=-1, keepdim=True
    )
    axis_angle = axis_angle * torch.linspace(
        0.0, 3.0, 64, dtype=torch.float64
    )[:, None]
    rotation = axis_angle_to_matrix(axis_angle)
    canonical = canonicalize_continuous_y_rotation(
        rotation, torch.zeros(64, dtype=torch.long)
    )

    identity = torch.eye(3, dtype=torch.float64)[None]
    relative = rotation.transpose(-1, -2) @ canonical
    local_y = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64)
    assert torch.allclose(
        canonical[:, :, 1], rotation[:, :, 1], atol=1e-10, rtol=1e-10
    )
    assert torch.allclose(
        canonical.transpose(-1, -2) @ canonical,
        identity.expand_as(canonical),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        torch.linalg.det(canonical),
        torch.ones(64, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )
    assert torch.allclose(
        relative @ local_y,
        local_y[None].expand(64, -1),
        atol=1e-10,
        rtol=1e-10,
    )

    asymmetric = canonicalize_continuous_y_rotation(
        rotation, torch.full((64,), 2, dtype=torch.long)
    )
    assert torch.equal(asymmetric, rotation)


def test_continuous_y_gauge_reference_axis_threshold_is_exact():
    dtype = torch.float64
    camera_x = torch.tensor((1.0, 0.0, 0.0), dtype=dtype)
    camera_z = torch.tensor((0.0, 0.0, 1.0), dtype=dtype)

    def rotation_with_y_axis(x_coordinate: float) -> torch.Tensor:
        y_axis = torch.tensor(
            (
                x_coordinate,
                math.sqrt(1.0 - x_coordinate * x_coordinate),
                0.0,
            ),
            dtype=dtype,
        )
        x_axis = camera_z
        z_axis = torch.linalg.cross(x_axis, y_axis)
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    def expected_gauge(
        rotation: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        y_axis = rotation[:, 1]
        x_axis = reference - torch.dot(reference, y_axis) * y_axis
        x_axis = x_axis / torch.linalg.vector_norm(x_axis)
        z_axis = torch.linalg.cross(x_axis, y_axis)
        z_axis = z_axis / torch.linalg.vector_norm(z_axis)
        x_axis = torch.linalg.cross(y_axis, z_axis)
        x_axis = x_axis / torch.linalg.vector_norm(x_axis)
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    x_coordinates = (0.899999, 0.90, 0.900001)
    rotation = torch.stack(
        [rotation_with_y_axis(value) for value in x_coordinates]
    )
    canonical = canonicalize_continuous_y_rotation(
        rotation, torch.zeros(3, dtype=torch.long)
    )

    assert abs(float(rotation[0, 0, 1])) < 0.90
    assert float(rotation[1, 0, 1]) == 0.90
    assert abs(float(rotation[2, 0, 1])) > 0.90
    assert torch.allclose(
        canonical[0],
        expected_gauge(rotation[0], camera_x),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        canonical[1],
        expected_gauge(rotation[1], camera_x),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.allclose(
        canonical[2],
        expected_gauge(rotation[2], camera_z),
        atol=1e-12,
        rtol=1e-12,
    )


def test_information_irls_reduces_a_metric_alignment_residual():
    torch.manual_seed(260907)
    canonical = torch.randn(2, 48, 3) * 0.25
    target_size = torch.tensor([[0.22, 0.31, 0.18], [0.30, 0.16, 0.25]])
    target_rotation = axis_angle_to_matrix(
        torch.tensor([[0.10, -0.07, 0.04], [-0.08, 0.05, 0.12]])
    )
    target_translation = torch.tensor([[0.03, -0.02, 0.80], [-0.04, 0.01, 0.95]])
    camera = torch.einsum(
        "bij,bkj->bki", target_rotation, canonical * target_size[:, None]
    ) + target_translation[:, None]
    initial_size = target_size * torch.tensor([1.08, 0.92, 1.05])
    solved = information_irls(
        canonical,
        camera,
        initial_size,
        torch.ones(2, 48),
        torch.ones(2, 48, 3),
        torch.ones(2, 48, 4),
        iterations=3,
        damping=1e-3,
        max_rotation_step_degrees=8.0,
        max_translation_step=0.015,
        max_log_size_step=0.12,
    )
    initial_camera = torch.einsum(
        "bij,bkj->bki",
        solved["initial_rotation"],
        canonical * initial_size[:, None],
    ) + solved["initial_translation"][:, None]
    final_camera = torch.einsum(
        "bij,bkj->bki",
        solved["rotation"],
        canonical * solved["size"][:, None],
    ) + solved["translation"][:, None]
    assert (final_camera - camera).square().mean() < (
        initial_camera - camera
    ).square().mean()
    assert solved["solver_gate"].shape == (2, 3)
    assert solved["solver_condition"].shape == (2, 3)


def test_information_irls_applies_the_documented_per_step_bounds():
    torch.manual_seed(260931)
    canonical = torch.randn(3, 40, 3) * 0.35
    camera = torch.randn(3, 40, 3) * 0.45 + torch.tensor([0.2, -0.1, 0.9])
    initial_size = torch.full((3, 3), 0.20)
    solved = information_irls(
        canonical,
        camera,
        initial_size,
        torch.ones(3, 40),
        torch.ones(3, 40, 3),
        torch.ones(3, 40, 4),
        iterations=1,
        damping=1e-3,
        max_rotation_step_degrees=8.0,
        max_translation_step=0.015,
        max_log_size_step=0.12,
    )
    relative = solved["rotation"] @ solved["initial_rotation"].transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0)
    rotation_step = torch.acos(cosine.clamp(-1.0, 1.0))
    translation_step = torch.linalg.vector_norm(
        solved["translation"] - solved["initial_translation"], dim=-1
    )
    log_size_step = (
        solved["size"].log() - initial_size.log()
    ).abs().amax(dim=-1)
    assert torch.all(rotation_step <= torch.deg2rad(torch.tensor(8.0)) + 1e-5)
    assert torch.all(translation_step <= 0.015 + 1e-6)
    assert torch.all(log_size_step <= 0.12 + 1e-6)


def test_joint_residual_jacobian_matches_left_increment_finite_difference():
    torch.manual_seed(260932)
    dtype = torch.float64
    canonical = torch.randn(2, 7, 3, dtype=dtype) * 0.25
    log_size = torch.tensor(
        [[-1.45, -1.20, -1.60], [-1.10, -1.55, -1.30]], dtype=dtype
    )
    size = log_size.exp()
    rotation = axis_angle_to_matrix(
        torch.tensor([[0.18, -0.11, 0.07], [-0.09, 0.14, 0.12]], dtype=dtype)
    )
    translation = torch.tensor(
        [[0.03, -0.02, 0.80], [-0.04, 0.01, 0.95]], dtype=dtype
    )
    camera = torch.randn(2, 7, 3, dtype=dtype) * 0.03
    metric = canonical * size[:, None]
    rotated = torch.einsum("bij,bkj->bki", rotation, metric)
    residual = rotated + translation[:, None] - camera
    rotation_jacobian = -_skew(rotated)
    translation_jacobian = torch.eye(3, dtype=dtype)[None, None].expand(
        2, 7, -1, -1
    )
    size_jacobian = rotation[:, None] * metric[:, :, None, :]
    jacobian = torch.cat(
        (rotation_jacobian, translation_jacobian, size_jacobian), dim=-1
    )

    direction = torch.randn(2, 9, dtype=dtype)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    )
    epsilon = 1e-6
    perturbed_rotation = axis_angle_to_matrix(
        epsilon * direction[:, :3]
    ) @ rotation
    perturbed_translation = translation + epsilon * direction[:, 3:6]
    perturbed_size = torch.exp(log_size + epsilon * direction[:, 6:9])
    perturbed_metric = canonical * perturbed_size[:, None]
    perturbed_residual = (
        torch.einsum(
            "bij,bkj->bki", perturbed_rotation, perturbed_metric
        )
        + perturbed_translation[:, None]
        - camera
    )

    finite_difference = (perturbed_residual - residual) / epsilon
    analytic = torch.einsum("bkri,bi->bkr", jacobian, direction)
    assert torch.allclose(finite_difference, analytic, atol=2e-6, rtol=2e-5)


def test_information_irls_has_finite_local_gradients_on_an_active_branch():
    torch.manual_seed(260933)
    canonical = (torch.randn(1, 24, 3) * 0.20).requires_grad_()
    initial_size = torch.tensor([[0.22, 0.27, 0.19]], requires_grad=True)
    target_rotation = axis_angle_to_matrix(
        torch.tensor([[0.04, -0.03, 0.02]])
    )
    target_translation = torch.tensor([[0.01, -0.02, 0.75]])
    camera = (
        torch.einsum(
            "bij,bkj->bki",
            target_rotation,
            canonical.detach() * initial_size.detach()[:, None],
        )
        + target_translation[:, None]
    )
    camera = camera + torch.randn_like(camera) * 0.002
    inlier = torch.full((1, 24), 0.85, requires_grad=True)
    precision = torch.full((1, 24, 3), 1.2, requires_grad=True)
    information = torch.full((1, 24, 4), 1.1, requires_grad=True)
    solved = information_irls(
        canonical,
        camera,
        initial_size,
        inlier,
        precision,
        information,
        iterations=1,
        damping=1e-3,
        max_rotation_step_degrees=8.0,
        max_translation_step=0.015,
        max_log_size_step=0.12,
    )
    loss = (
        solved["rotation"].square().sum()
        + solved["translation"].square().sum()
        + solved["size"].square().sum()
    )
    loss.backward()
    for value in (canonical, initial_size, inlier, precision, information):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_stage2_has_one_finite_three_step_irls_endpoint():
    model = Stage2PoseInformationField(
        token_width=48,
        anchor_count=8,
        relation_layers=1,
        heads=6,
        dropout=0.0,
    ).eval()
    points = torch.randn(2, 24, 3) * 0.05 + torch.tensor([0.0, 0.0, 0.8])
    center = points.mean(dim=1)
    normals, observation = estimate_camera_observation_features(points, center, k=8)
    with torch.no_grad():
        output = model(
            points,
            normals,
            observation,
            alignment_center=center,
            continuous_symmetry=torch.tensor([False, True]),
        )
    assert output["pose"].shape == (2, 9)
    assert output["size"].shape == (2, 3)
    assert output["size"].dtype == torch.float32
    assert output["information"].shape == (2, 8, 4)
    assert output["solver_weights"].shape == (2, 8)
    assert output["solver_singular_values"].shape == (2, 3)
    assert output["initial_rotation"].shape == (2, 3, 3)
    assert output["initial_translation"].shape == (2, 3)
    assert output["solver_gate"].shape == (2, 3)
    assert output["solver_condition"].shape == (2, 3)
    assert output["solver_residual_rms"].shape == (2, 3)
    assert not any(
        "class" in name or "category" in name
        for name, _ in model.named_parameters()
    )
    assert all(
        torch.isfinite(value).all()
        for value in output.values()
        if torch.is_tensor(value)
    )


def test_unknown_endpoint_metadata_is_rejected():
    architecture = {
        "observation_feature_dim": 14,
        "token_width": 48,
        "anchor_count": 8,
        "relation_layers": 1,
        "heads": 6,
        "dropout": 0.0,
        "canonical_range": 0.65,
        "size_min": 0.01,
        "size_max": 0.80,
        "size_init": 0.20,
        "unsupported_endpoint_option": 3,
    }
    try:
        Stage2PoseInformationField.from_architecture(architecture)
    except ValueError as error:
        assert "unsupported Stage2 architecture fields" in str(error)
    else:
        raise AssertionError("unknown endpoint metadata must not be accepted")


def test_stage2_rejects_solver_parameters_outside_the_proved_domain():
    invalid_parameters = (
        {"irls_iterations": -1},
        {"irls_iterations": 1.5},
        {"irls_damping": 0.0},
        {"max_rotation_step_degrees": 0.0},
        {"max_translation_step": 0.0},
        {"max_log_size_step": 0.0},
        {"irls_damping": float("nan")},
    )
    for parameters in invalid_parameters:
        try:
            Stage2PoseInformationField(
                token_width=48,
                anchor_count=8,
                relation_layers=1,
                heads=6,
                **parameters,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid solver parameters were accepted: {parameters}")


def test_information_irls_rejects_nonpositive_damping():
    canonical = torch.zeros(1, 4, 3)
    inputs = {
        "canonical": canonical,
        "camera_xyz": canonical.clone(),
        "initial_size": torch.full((1, 3), 0.2),
        "inlier_probability": torch.ones(1, 4),
        "axis_precision": torch.ones(1, 4, 3),
        "information": torch.ones(1, 4, 4),
        "iterations": 1,
        "damping": 0.0,
        "max_rotation_step_degrees": 8.0,
        "max_translation_step": 0.015,
        "max_log_size_step": 0.12,
    }
    try:
        information_irls(**inputs)
    except ValueError as error:
        assert "damping" in str(error).lower()
    else:
        raise AssertionError("information_irls accepted zero damping")
