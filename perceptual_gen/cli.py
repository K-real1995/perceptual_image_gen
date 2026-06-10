import argparse
import sys
from pathlib import Path

from perceptual_gen.config import (
    ALL_LAYERS_TOKEN,
    IMAGE_EXTENSIONS,
    MODE_CONTENT,
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
            "Generate images via VGG19 optimization: content reconstruction "
            "or texture synthesis."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        "-i",
        help="Path to a single source image.",
    )
    input_group.add_argument(
        "--input-dir",
        help="Directory with images for batch texture runs (2-3 textures).",
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
        help="Optimization steps. Defaults: 500 (content), 2500 (texture).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate. Defaults: 0.2 (content), 0.05 (texture).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=256,
        help="Maximum size of the longest image side after resize.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=None,
        help="Total variation weight. Defaults: 0.01 (content), 10000 (texture).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated image initialization.",
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


def config_from_args(args: argparse.Namespace, input_path: Path) -> GenerationConfig:
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input_dir and args.mode != MODE_TEXTURE:
        print(
            "Error: --input-dir is intended for texture batch runs. Use --mode texture.",
            file=sys.stderr,
        )
        return 1

    try:
        input_paths = _collect_input_paths(args)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    all_results: list[tuple[Path, list]] = []
    try:
        for index, input_path in enumerate(input_paths):
            if len(input_paths) == 1:
                output_dir = args.output_dir
            else:
                output_dir = str(Path(args.output_dir) / input_path.stem)

            config = config_from_args(args, input_path)
            results = run_pipeline(config)
            all_results.append((input_path, results))
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for input_path, results in all_results:
        for result in results:
            print(
                f"[{input_path.name}][{result.layer}] loss={result.final_loss:.4f}, "
                f"time={result.elapsed_sec:.1f}s, result={result.result_path}"
            )
    return 0
