import argparse
import sys
from pathlib import Path

from perceptual_gen.config import (
    ALL_LAYERS_TOKEN,
    IMAGE_EXTENSIONS,
    MODE_CONTENT,
    MODE_STYLE_TRANSFER,
    MODE_TEXTURE,
    SUPPORTED_LAYERS,
    SUPPORTED_MODES,
    GenerationConfig,
)
from perceptual_gen.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    content_layers = ", ".join([*SUPPORTED_LAYERS, ALL_LAYERS_TOKEN])
    modes = ", ".join(SUPPORTED_MODES)
    parser = argparse.ArgumentParser(
        description=(
            "Generate images via VGG19 optimization: content reconstruction, "
            "texture synthesis, or neural style transfer."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input",
        "-i",
        help="Path to a single source image (content/texture modes).",
    )
    input_group.add_argument(
        "--input-dir",
        help="Directory with images for batch texture runs.",
    )
    parser.add_argument(
        "--content",
        help="Content image for style-transfer mode.",
    )
    parser.add_argument(
        "--style",
        help="Style image for style-transfer mode.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=SUPPORTED_MODES,
        default=MODE_CONTENT,
        help=f"Generation mode. Supported: {modes}.",
    )
    parser.add_argument(
        "--layer",
        "-l",
        default=None,
        help=(
            f"Content mode only: VGG19 layer or '{ALL_LAYERS_TOKEN}'. "
            f"Supported: {content_layers}."
        ),
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=None,
        help="Optimization steps. Mode-specific defaults apply.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate. Mode-specific defaults apply.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Maximum size of the longest image side after resize.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=None,
        help="Total variation weight. Mode-specific defaults apply.",
    )
    parser.add_argument(
        "--style-weight",
        type=float,
        default=None,
        help="Style loss weight for style-transfer mode (default: 100).",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=None,
        help="Content loss weight for style-transfer mode (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise initialization in content/texture modes.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate snapshots every N steps. 0 disables snapshots.",
    )
    return parser


def _collect_input_paths(args: argparse.Namespace) -> list[Path]:
    if args.input:
        return [Path(args.input)]

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No images found in {input_dir}. Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return images


def _validate_args(args: argparse.Namespace) -> str | None:
    if args.mode == MODE_STYLE_TRANSFER:
        if not args.content or not args.style:
            return "Style-transfer mode requires --content and --style."
        return None

    if not args.input and not args.input_dir:
        return f"{args.mode} mode requires --input or --input-dir."

    if args.input_dir and args.mode != MODE_TEXTURE:
        return "--input-dir is intended for texture batch runs. Use --mode texture."

    return None


def config_from_args(
    args: argparse.Namespace,
    *,
    input_path: Path,
    style_path: Path | None = None,
) -> GenerationConfig:
    return GenerationConfig.with_mode_defaults(
        input_path=str(input_path),
        output_dir=args.output_dir,
        mode=args.mode,
        layer=args.layer,
        steps=args.steps,
        learning_rate=args.lr,
        max_dim=args.max_dim,
        tv_weight=args.tv_weight,
        seed=args.seed,
        save_every=args.save_every,
        style_path=str(style_path) if style_path else args.style,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    validation_error = _validate_args(args)
    if validation_error:
        print(f"Error: {validation_error}", file=sys.stderr)
        return 1

    all_results: list[tuple[str, list]] = []

    try:
        if args.mode == MODE_STYLE_TRANSFER:
            content_path = Path(args.content)
            style_path = Path(args.style)
            config = config_from_args(
                args,
                input_path=content_path,
                style_path=style_path,
            )
            results = run_pipeline(config)
            all_results.append((content_path.name, results))
        else:
            input_paths = _collect_input_paths(args)
            for input_path in input_paths:
                if len(input_paths) == 1:
                    output_dir = args.output_dir
                else:
                    output_dir = str(Path(args.output_dir) / input_path.stem)

                config = config_from_args(args, input_path=input_path)
                config.output_dir = output_dir
                results = run_pipeline(config)
                all_results.append((input_path.name, results))
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for label, results in all_results:
        for result in results:
            print(
                f"[{label}][{result.layer}] loss={result.final_loss:.4f}, "
                f"time={result.elapsed_sec:.1f}s, result={result.result_path}"
            )
    return 0
