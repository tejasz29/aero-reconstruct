"""STEP 1 — CLI smoke tests: version / init / doctor behave as documented."""

from src import cli


def test_version_command(capsys):
    assert cli.main(["version"]) == 0
    assert cli.VERSION in capsys.readouterr().out


def test_init_is_idempotent(capsys):
    assert cli.main(["init"]) == 0
    assert "initialised" in capsys.readouterr().out


def test_doctor_reports_layout_ok(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "directories  : OK" in out
    assert "config       : OK" in out


def test_doctor_fails_on_broken_layout(tmp_path, monkeypatch, capsys):
    """Point the CLI at an empty dir -> doctor must exit non-zero."""
    import src.common.paths as paths_mod

    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    assert cli.main(["doctor"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_build_parser_lists_all_stage_subcommands():
    help_text = cli.build_parser().format_help()
    for subcommand in ("extract-frames", "select-keyframes", "calibrate",
                       "reconstruct-poses", "show-trajectory"):
        assert subcommand in help_text


def test_reconstruct_poses_parser_exposes_expected_flags():
    parser = cli.build_parser()
    argv = ["reconstruct-poses", "--frames-dir", "frames",
            "--keyframes-file", "kf.csv", "--camera", "cam.yaml",
            "--output-dir", "out"]
    args = parser.parse_args(argv)
    assert args.frames_dir == "frames"
    assert args.keyframes_file == "kf.csv"
    assert args.camera == "cam.yaml"
    assert args.output_dir == "out"


def test_show_trajectory_parser_exposes_expected_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["show-trajectory", "--poses-csv", "poses.csv",
                              "--output-dir", "figs", "--topdown", "xz"])
    assert args.poses_csv == "poses.csv"
    assert args.output_dir == "figs"
    assert args.topdown == "xz"
