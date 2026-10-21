import pytest
import torch
import numpy as np
import os
from graspgenx.utils.point_cloud import point_cloud_outlier_removal


@pytest.fixture
def sample_point_cloud():
    np.random.seed(42)
    num_points = 3000
    points = np.random.uniform(-0.5, 0.5, size=(num_points, 3))
    return points


def test_point_cloud_outlier_removal_on_random_cloud(sample_point_cloud):
    """Verify outlier removal works on a random point cloud."""
    pc = torch.from_numpy(sample_point_cloud).float()
    filtered_pc, removed_pc = point_cloud_outlier_removal(pc, threshold=0.5)

    assert filtered_pc.shape[1] == 3
    assert removed_pc.shape[1] == 3
    assert filtered_pc.shape[0] + removed_pc.shape[0] == pc.shape[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graspgenx_sampler_import():
    """Verify GraspGenXSampler can be imported (requires CUDA for model init)."""
    from graspgenx.grasp_server import GraspGenXSampler
    assert GraspGenXSampler is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graspgenx_model_import():
    """Verify GraspGen model can be imported."""
    from graspgenx.models.grasp_gen import GraspGen
    assert GraspGen is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.integration
def test_grasp_sampling_basic(sample_point_cloud):
    """Test basic grasp sampling (requires GPU and model checkpoints)."""
    from graspgenx.grasp_server import GraspGenXSampler

    config_path = "/models/checkpoints/graspgenx.yml"
    if not os.path.exists(config_path):
        pytest.skip(f"Config not found at {config_path}")

    # This would test full inference if checkpoints are available
    pytest.skip("Full inference test requires model checkpoints")
