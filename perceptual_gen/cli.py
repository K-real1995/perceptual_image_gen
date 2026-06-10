import argparse
import sys

from perceptual_gen.config import ALL_LAYERS_TOKEN, SUPPORTED_LAYERS, GenerationConfig
from perceptual_gen.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    supported = ", ".join([*SUPPORTED_LAYERS, ALL_LAYERS_TOKEN])
    parser = argparse.ArgumentParser(
        description="Generate images by optimizing perceptual loss with VGG19 features.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the source image.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--layer",
        "-l",
        default="block4_conv1",
        help=f"VGG19 layer to optimize against. Supported: {supported}.",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=500,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Adam learning rate.",
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
        default=0.01,
        help="Total variation regularization weight.",
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


def config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        input_path=args.input,
        output_dir=args.output_dir,
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
    config = config_from_args(args)

    try:
        results = run_pipeline(config)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for result in results:
        print(
            f"[{result.layer}] loss={result.final_loss:.4f}, "
            f"time={result.elapsed_sec:.1f}s, result={result.result_path}"
        )
    return 0
