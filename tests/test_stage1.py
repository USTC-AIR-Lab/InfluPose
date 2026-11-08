import torch

from stage1.model import Stage1CompleteCloudRAE


def test_stage1_small_forward_is_category_free_and_finite():
    model = Stage1CompleteCloudRAE(
        hidden_dim=64,
        base_feature_dim=32,
        directional_feature_dim=32,
        reconstruction_points=8,
        query_dim=8,
        transformer_layers=1,
        transformer_heads=4,
        geometric_stage_dim_1=16,
        geometric_stage_dim_2=24,
        geometric_stage_dim_3=32,
        geometric_sample_count_1=8,
        geometric_sample_count_2=4,
        geometric_sample_count_3=2,
        geometric_radius_1=0.5,
        geometric_radius_2=0.75,
        geometric_radius_3=1.0,
        geometric_neighbors=8,
        geometric_local_heads=4,
        directional_token_dim=32,
        directional_token_count=8,
        directional_transformer_layers=1,
        directional_transformer_heads=4,
        dropout=0.0,
    ).eval()
    points = torch.randn(2, 16, 3) * 0.05 + torch.tensor([0.0, 0.0, 0.8])
    normals = torch.nn.functional.normalize(torch.randn_like(points), dim=-1)
    center = points.mean(dim=1)
    with torch.no_grad():
        output = model(
            points,
            alignment_center=center,
            normals=normals,
        )
    assert output["latent_mean"].shape == (2, 128)
    assert output["pose"].shape == (2, 9)
    assert output["size"].shape == (2, 3)
    assert output["reconstructed_points"].shape == (2, 8, 3)
    assert all(
        torch.isfinite(value).all()
        for value in output.values()
        if torch.is_tensor(value)
    )
    assert model.base_encoder.class_embedding is None
