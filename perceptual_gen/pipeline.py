import json
import time
from pathlib import Path

import tensorflow as tf

from perceptual_gen.config import (
    MODE_CONTENT,
    MODE_STYLE_TRANSFER,
    MODE_TEXTURE,
    GenerationConfig,
    RunResult,
)
from perceptual_gen.image_io import load_image, save_image
from perceptual_gen.loss import build_perceptual_loss
from perceptual_gen.optimizer import optimize_image
from perceptual_gen.style_content_extractor import (
    DEFAULT_CONTENT_LAYERS,
    StyleAndContentExtractor,
)
from perceptual_gen.style_content_loss import build_style_content_loss
from perceptual_gen.style_extractor import DEFAULT_STYLE_LAYERS, StyleExtractor
from perceptual_gen.style_loss import build_style_loss
from perceptual_gen.visualization import save_comparison, save_style_transfer_comparison
from perceptual_gen.vgg_extractor import FeatureExtractor


def run_content_layer(
    content_image: tf.Tensor,
    config: GenerationConfig,
    layer: str,
    output_dir: str | Path,
) -> RunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor([layer])
    target_features = extractor(content_image)
    loss_func = build_perceptual_loss(extractor, target_features, config.tv_weight)

    snapshot_dir = output_dir / "steps" if config.save_every > 0 else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    generated_image, loss_history = optimize_image(
        content_image=content_image,
        loss_func=loss_func,
        steps=config.steps,
        learning_rate=config.learning_rate,
        seed=config.seed,
        save_every=config.save_every,
        snapshot_dir=snapshot_dir,
    )
    elapsed_sec = time.perf_counter() - started

    result_path = output_dir / "result.png"
    comparison_path = output_dir / "comparison.png"
    save_image(generated_image, result_path)
    save_comparison(
        content_image,
        generated_image,
        layer,
        comparison_path,
        mode=MODE_CONTENT,
    )

    run_result = RunResult(
        layer=layer,
        final_loss=loss_history[-1] if loss_history else 0.0,
        elapsed_sec=elapsed_sec,
        result_path=str(result_path),
        comparison_path=str(comparison_path),
        loss_history=loss_history,
    )

    config_payload = {
        **config.to_dict(),
        "layer": layer,
        "final_loss": run_result.final_loss,
        "elapsed_sec": run_result.elapsed_sec,
        "loss_history": run_result.loss_history,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    return run_result


def run_texture(
    style_image: tf.Tensor,
    config: GenerationConfig,
    output_dir: str | Path,
) -> RunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = StyleExtractor(list(DEFAULT_STYLE_LAYERS))
    style_targets = extractor(style_image)
    loss_func = build_style_loss(extractor, style_targets, config.tv_weight)

    snapshot_dir = output_dir / "steps" if config.save_every > 0 else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    generated_image, loss_history = optimize_image(
        content_image=style_image,
        loss_func=loss_func,
        steps=config.steps,
        learning_rate=config.learning_rate,
        seed=config.seed,
        save_every=config.save_every,
        snapshot_dir=snapshot_dir,
    )
    elapsed_sec = time.perf_counter() - started

    result_path = output_dir / "result.png"
    comparison_path = output_dir / "comparison.png"
    save_image(generated_image, result_path)
    save_comparison(
        style_image,
        generated_image,
        "style",
        comparison_path,
        mode=MODE_TEXTURE,
    )

    run_result = RunResult(
        layer="style",
        final_loss=loss_history[-1] if loss_history else 0.0,
        elapsed_sec=elapsed_sec,
        result_path=str(result_path),
        comparison_path=str(comparison_path),
        loss_history=loss_history,
    )

    config_payload = {
        **config.to_dict(),
        "style_layers": list(DEFAULT_STYLE_LAYERS),
        "final_loss": run_result.final_loss,
        "elapsed_sec": run_result.elapsed_sec,
        "loss_history": run_result.loss_history,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    return run_result


def run_style_transfer(
    content_image: tf.Tensor,
    style_image: tf.Tensor,
    config: GenerationConfig,
    output_dir: str | Path,
) -> RunResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = StyleAndContentExtractor(
        style_layers=list(DEFAULT_STYLE_LAYERS),
        content_layers=list(DEFAULT_CONTENT_LAYERS),
    )
    style_features = extractor(style_image)
    content_features = extractor(content_image)
    loss_func = build_style_content_loss(
        extractor=extractor,
        style_targets=style_features["style"],
        content_targets=content_features["content"],
        style_weight=config.style_weight,
        content_weight=config.content_weight,
        tv_weight=config.tv_weight,
    )

    snapshot_dir = output_dir / "steps" if config.save_every > 0 else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    generated_image, loss_history = optimize_image(
        content_image=content_image,
        loss_func=loss_func,
        steps=config.steps,
        learning_rate=config.learning_rate,
        seed=config.seed,
        save_every=config.save_every,
        snapshot_dir=snapshot_dir,
        init_image=content_image,
    )
    elapsed_sec = time.perf_counter() - started

    result_path = output_dir / "result.png"
    comparison_path = output_dir / "comparison.png"
    save_image(generated_image, result_path)
    save_style_transfer_comparison(
        content_image,
        style_image,
        generated_image,
        comparison_path,
    )

    run_result = RunResult(
        layer="style-transfer",
        final_loss=loss_history[-1] if loss_history else 0.0,
        elapsed_sec=elapsed_sec,
        result_path=str(result_path),
        comparison_path=str(comparison_path),
        loss_history=loss_history,
    )

    config_payload = {
        **config.to_dict(),
        "style_layers": list(DEFAULT_STYLE_LAYERS),
        "content_layers": list(DEFAULT_CONTENT_LAYERS),
        "final_loss": run_result.final_loss,
        "elapsed_sec": run_result.elapsed_sec,
        "loss_history": run_result.loss_history,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2)

    return run_result


def run_pipeline(config: GenerationConfig) -> list[RunResult]:
    config.validate()

    if config.mode == MODE_STYLE_TRANSFER:
        style_path = config.style_path
        if style_path is None:
            raise ValueError("Style-transfer mode requires --style")
        content_image = load_image(config.input_path, max_dim=config.max_dim)
        style_image = load_image(style_path, max_dim=config.max_dim)
        return [
            run_style_transfer(
                content_image,
                style_image,
                config,
                config.output_dir,
            )
        ]

    source_image = load_image(config.input_path, max_dim=config.max_dim)

    if config.mode == MODE_TEXTURE:
        return [run_texture(source_image, config, config.output_dir)]

    layers = config.resolved_layers()
    results: list[RunResult] = []

    if len(layers) == 1:
        results.append(
            run_content_layer(source_image, config, layers[0], config.output_dir)
        )
        return results

    base_output_dir = Path(config.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        layer_output_dir = base_output_dir / layer
        results.append(
            run_content_layer(source_image, config, layer, layer_output_dir)
        )

    summary = {
        "input_path": config.input_path,
        "mode": config.mode,
        "layers": [result.to_dict() for result in results],
    }
    with (base_output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return results
