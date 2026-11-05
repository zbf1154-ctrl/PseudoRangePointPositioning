from spp.cli import main


def test_cli_info(capsys):
    assert main(["info"]) == 0
    output = capsys.readouterr().out
    assert "版本" in output
    assert "ABMF" not in output  # info 只显示路径，不解析数据
    assert "igs20865.sp3" in output


def test_cli_run_small(tmp_path, capsys):
    exit_code = main(
        [
            "run",
            "--start",
            "2020-01-03 00:00:00",
            "--end",
            "2020-01-03 00:01:00",
            "--out",
            str(tmp_path),
            "--prefix",
            "cli",
            "--no-plot",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "cli_solutions.csv").is_file()
    assert (tmp_path / "cli_summary.json").is_file()
    output = capsys.readouterr().out
    assert "ABMF" in output
    assert "有效/总历元" in output


def test_cli_without_command(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_bad_file(tmp_path, capsys):
    exit_code = main(
        [
            "run",
            "--obs",
            str(tmp_path / "missing.20o"),
            "--out",
            str(tmp_path),
            "--no-plot",
            "--quiet",
        ]
    )
    assert exit_code == 1
    assert "不存在" in capsys.readouterr().err
