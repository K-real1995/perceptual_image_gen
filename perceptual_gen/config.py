from dataclasses import asdict, dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


def _resolve_optional(value: T | None, default: T) -> T:
    return default if value is None else value

MODE_CONTENT = "content"
MODE_TEXTURE = "texture"
MODE_STYLE_TRANSFER = "style-transfer"
SUPPORTED_MODES = (MODE_CONTENT, MODE_TEXTURE, MODE_STYLE_TRANSFER)

SUPPORTED_LAYERS = ("block2_conv2", "block4_conv1", "block5_conv3")
ALL_LAYERS_TOKEN = "all"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ModeDefaults:
    steps: int
    learning_rate: float
    tv_weight: float
    layer: str
    max_dim: int = 256


CONTENT_DEFAULTS = ModeDefaults(
    steps=500,
    learning_rate=0.2,
    tv_weight=0.01,
    layer="block4_conv1",
    max_dim=256,
)

TEXTURE_DEFAULTS = ModeDefaults(
    steps=2500,
    learning_rate=0.05,
    tv_weight=1e4,
    layer="style",
    max_dim=256,
)

STYLE_TRANSFER_DEFAULTS = ModeDefaults(
    steps=1000,
    learning_rate=0.02,
    tv_weight=0.1,
    layer="style-transfer",
    max_dim=512,
)

DEFAULT_STYLE_WEIGHT = 100.0
DEFAULT_CONTENT_WEIGHT = 5.0


def _defaults_for_mode(mode: str) -> ModeDefaults:
    if mode == MODE_TEXTURE:
        return TEXTURE_DEFAULTS
    if mode == MODE_STYLE_TRANSFER:
        return STYLE_TRANSFER_DEFAULTS
    return CONTENT_DEFAULTS


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
    style_path: str | None = None
    style_weight: float = DEFAULT_STYLE_WEIGHT
    content_weight: float = DEFAULT_CONTENT_WEIGHT

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
        max_dim: int | None = None,
        tv_weight: float | None = None,
        seed: int = 42,
        save_every: int = 0,
        style_path: str | None = None,
        style_weight: float | None = None,
        content_weight: float | None = None,
    ) -> "GenerationConfig":
        defaults = _defaults_for_mode(mode)
        return cls(
            input_path=input_path,
            output_dir=output_dir,
            mode=mode,
            layer=_resolve_optional(layer, defaults.layer),
            steps=_resolve_optional(steps, defaults.steps),
            learning_rate=_resolve_optional(learning_rate, defaults.learning_rate),
            max_dim=_resolve_optional(max_dim, defaults.max_dim),
            tv_weight=_resolve_optional(tv_weight, defaults.tv_weight),
            seed=seed,
            save_every=save_every,
            style_path=style_path,
            style_weight=_resolve_optional(style_weight, DEFAULT_STYLE_WEIGHT),
            content_weight=_resolve_optional(content_weight, DEFAULT_CONTENT_WEIGHT),
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
        if self.mode == MODE_STYLE_TRANSFER:
            if not self.style_path:
                raise ValueError("Style-transfer mode requires --style")
            return
        if self.mode == MODE_CONTENT and self.layer != ALL_LAYERS_TOKEN:
            if self.layer not in SUPPORTED_LAYERS:
                supported = ", ".join([*SUPPORTED_LAYERS, ALL_LAYERS_TOKEN])
                raise ValueError(
                    f"Unsupported layer '{self.layer}'. Supported values: {supported}"
                )

    def resolved_layers(self) -> list[str]:
        if self.mode in (MODE_TEXTURE, MODE_STYLE_TRANSFER):
            return [self.layer]
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
