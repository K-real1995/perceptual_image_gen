from pathlib import Path

from perceptual_gen.cli import main


def test_cli_smoke_run(sample_image_path: Path, tmp_path: Path):
    output_dir = tmp_path / "smoke_run"
    exit_code = main(
        [
            "--input",
            str(sample_image_path),
            "--output-dir",
            str(output_dir),
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
