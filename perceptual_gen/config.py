from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


def _resolve_optional(value: T | None, default: T) -> T:
    return default if value is None else value

MODE_CONTENT = "content"
MODE_TEXTURE = "texture"
SUPPORTED_MODES = (MODE_CONTENT, MODE_TEXTURE)

SUPPORTED_LAYERS = ("block2_conv2", "block4_conv1", "block5_conv3")
ALL_LAYERS_TOKEN = "all"

@dataclass(frozen=True)
class ModeDefaults:
    steps: int
    learning_rate: float
    tv_weight: float
    layer: str


CONTENT_DEFAULTS = ModeDefaults(
    steps=500,
    learning_rate=0.2,
    tv_weight=0.01,
    layer="block4_conv1",
)

TEXTURE_DEFAULTS = ModeDefaults(
    steps=2500,
    learning_rate=0.05,
    tv_weight=1e4,
    layer="style",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class GenerationConfig:
    input_path: str
    output_dir: str
    mode: str = MODE_CONTENT
    layer: str = "block4_conv1"
    steps: int = 500
    learning_rate: float = 0.2
    max_dim: int = 256
    tv_weight: float = 0.01
    seed: int = 42
    save_every: int = 0

    @classmethod
    def with_mode_defaults(
        cls,
        *,
        input_path: str,
        output_dir: str,
        mode: str = MODE_CONTENT,
        layer: str | None = None,
        steps: int | None = None,
        learning_rate: float | None = None,
        max_dim: int = 256,
        tv_weight: float | None = None,
        seed: int = 42,
        save_every: int = 0,
    ) -> "GenerationConfig":
        defaults = TEXTURE_DEFAULTS if mode == MODE_TEXTURE else CONTENT_DEFAULTS
        return cls(
            input_path=input_path,
            output_dir=output_dir,
            mode=mode,
            layer=_resolve_optional(layer, defaults.layer),
            steps=_resolve_optional(steps, defaults.steps),
            learning_rate=_resolve_optional(learning_rate, defaults.learning_rate),
            max_dim=max_dim,
            tv_weight=_resolve_optional(tv_weight, defaults.tv_weight),
            seed=seed,
            save_every=save_every,
        )

    def validate(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            supported = ", ".join(SUPPORTED_MODES)
            raise ValueError(f"Unsupported mode '{self.mode}'. Supported values: {supported}")
        if self.steps <= 0:
            raise ValueError("--steps must be a positive integer")
        if self.max_dim <= 0:
            raise ValueError("--max-dim must be a positive integer")
        if self.save_every < 0:
            raise ValueError("--save-every must be >= 0")
        if self.mode == MODE_CONTENT and self.layer != ALL_LAYERS_TOKEN:
            if self.layer not in SUPPORTED_LAYERS:
                supported = ", ".join([*SUPPORTED_LAYERS, ALL_LAYERS_TOKEN])
                raise ValueError(
                    f"Unsupported layer '{self.layer}'. Supported values: {supported}"
                )

    def resolved_layers(self) -> list[str]:
        if self.mode == MODE_TEXTURE:
            return ["style"]
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
