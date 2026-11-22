from pathlib import Path

from gui import build_parser, settings_from


def test_defaults_and_overrides():
    args = build_parser().parse_args([])
    settings = settings_from(args)
    assert settings.base_path == Path("base.json") and settings.clusters_path == Path("clusters.json")
    assert (args.host, args.port, settings.use_gpu) == ("127.0.0.1", 8000, False)

    custom = settings_from(build_parser().parse_args(["--base", "x.json", "--gpu", "--faces-dir", "ludzie", "--port", "9000"]))
    assert custom.base_path == Path("x.json") and custom.use_gpu and custom.faces_dir == "ludzie"


def test_deepfake_is_on_unless_disabled():
    assert settings_from(build_parser().parse_args([])).deepfake is True
    assert settings_from(build_parser().parse_args(["--no-deepfake"])).deepfake is False
