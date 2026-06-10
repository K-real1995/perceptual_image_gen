from pathlib import Path

import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / "sample.jpg"
    if not path.exists():
        pytest.skip("data/sample.jpg is required for integration tests")
    return path


@pytest.fixture(scope="session")
def sample_image_tensor(sample_image_path: Path) -> tf.Tensor:
    from perceptual_gen.image_io import load_image

    return load_image(sample_image_path, max_dim=128)
