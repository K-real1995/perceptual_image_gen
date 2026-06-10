from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptual_gen.config import MODE_TEXTURE
from perceptual_gen.image_io import ImageTensor, tensor_to_numpy


def _to_numpy(image: ImageTensor | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return tensor_to_numpy(image)


def save_comparison(
    original: ImageTensor | np.ndarray,
    generated: ImageTensor | np.ndarray,
    label: str,
    path: str | Path,
    mode: str = "content",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_array = _to_numpy(original)
    generated_array = _to_numpy(generated)

    if mode == MODE_TEXTURE:
        left_title = "Style reference"
        right_title = "Generated texture"
    else:
        left_title = "Original"
        right_title = f"Generated ({label})"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.clip(original_array, 0.0, 1.0))
    axes[0].set_title(left_title)
    axes[0].axis("off")

    axes[1].imshow(np.clip(generated_array, 0.0, 1.0))
    axes[1].set_title(right_title)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
