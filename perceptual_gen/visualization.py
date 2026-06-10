from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptual_gen.image_io import ImageTensor, tensor_to_numpy


def _to_numpy(image: ImageTensor | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return tensor_to_numpy(image)


def save_comparison(
    original: ImageTensor | np.ndarray,
    generated: ImageTensor | np.ndarray,
    layer: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    original_array = _to_numpy(original)
    generated_array = _to_numpy(generated)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.clip(original_array, 0.0, 1.0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(generated_array, 0.0, 1.0))
    axes[1].set_title(f"Generated ({layer})")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
