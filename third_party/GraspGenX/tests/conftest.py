import pytest
import torch
import numpy as np


def pytest_addoption(parser):
    parser.addoption(
        "--mesh",
        action="store",
        default=None,
        help="Path to .obj mesh file (absolute or relative to repo root). "
             "Defaults to assets provided in the repo.",
    )


@pytest.fixture
def device():
    """Provide the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_point_cloud():
    """Provide a sample point cloud for testing."""
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    return points


@pytest.fixture
def sample_rotation_matrix():
    """Provide a sample rotation matrix (90 degrees around z-axis)."""
    matrix = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return matrix


@pytest.fixture
def sample_pose():
    """Provide a sample 4x4 pose matrix."""
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    return pose
