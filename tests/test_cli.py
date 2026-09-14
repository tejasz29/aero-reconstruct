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
