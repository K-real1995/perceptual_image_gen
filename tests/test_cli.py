from pathlib import Path

from perceptual_gen.cli import main


def test_cli_content_smoke_run(sample_image_path: Path, tmp_path: Path):
    output_dir = tmp_path / "smoke_content"
    exit_code = main(
        [
            "--input",
            str(sample_image_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "content",
            "--steps",
            "2",
            "--layer",
            "block2_conv2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "result.png").exists()
    assert (output_dir / "comparison.png").exists()
    assert (output_dir / "config.json").exists()


def test_cli_texture_smoke_run(sample_image_path: Path, tmp_path: Path):
    output_dir = tmp_path / "smoke_texture"
    exit_code = main(
        [
            "--input",
            str(sample_image_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "texture",
            "--steps",
            "2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "result.png").exists()
    assert (output_dir / "comparison.png").exists()
    assert (output_dir / "config.json").exists()
