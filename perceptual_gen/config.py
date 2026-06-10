from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_LAYERS = ("block2_conv2", "block4_conv1", "block5_conv3")
ALL_LAYERS_TOKEN = "all"


@dataclass
class GenerationConfig:
    input_path: str
    output_dir: str
    layer: str = "block4_conv1"
    steps: int = 500
    learning_rate: float = 0.2
    max_dim: int = 256
    tv_weight: float = 0.01
    seed: int = 42
    save_every: int = 0

    def validate(self) -> None:
        if self.steps <= 0:
            raise ValueError("--steps must be a positive integer")
        if self.max_dim <= 0:
            raise ValueError("--max-dim must be a positive integer")
        if self.save_every < 0:
            raise ValueError("--save-every must be >= 0")
        if self.layer != ALL_LAYERS_TOKEN and self.layer not in SUPPORTED_LAYERS:
            supported = ", ".join([*SUPPORTED_LAYERS, ALL_LAYERS_TOKEN])
            raise ValueError(
                f"Unsupported layer '{self.layer}'. Supported values: {supported}"
            )

    def resolved_layers(self) -> list[str]:
        if self.layer == ALL_LAYERS_TOKEN:
            return list(SUPPORTED_LAYERS)
        return [self.layer]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    layer: str
    final_loss: float
    elapsed_sec: float
    result_path: str
    comparison_path: str
    loss_history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
