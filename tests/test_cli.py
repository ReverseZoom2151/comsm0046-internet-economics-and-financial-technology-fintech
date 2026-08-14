"""The command line interface runs each strand and writes nothing by default."""

from __future__ import annotations

import pytest

from fintech.cli import build_parser, main


class TestParser:
    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_an_unknown_command_is_refused(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["nonsense"])

    def test_an_unknown_market_scenario_is_refused(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["market", "nonsense"])

    def test_the_market_seed_has_a_default(self):
        assert build_parser().parse_args(["market"]).seed == 100


class TestDatasetsCommand:
    def test_it_lists_both_datasets(self, capsys):
        assert main(["datasets"]) == 0
        out = capsys.readouterr().out
        assert "data1" in out and "data2" in out


class TestStatsCommand:
    def test_it_names_the_test_it_chose(self, capsys):
        assert main(["stats", "data2"]) == 0
        out = capsys.readouterr().out
        assert "Kruskal-Wallis" in out

    def test_dataset_one_routes_to_anova(self, capsys):
        main(["stats", "data1"])
        assert "ANOVA" in capsys.readouterr().out

    def test_it_runs_both_by_default(self, capsys):
        main(["stats"])
        out = capsys.readouterr().out
        assert "data1" in out and "data2" in out

    def test_nothing_is_written_unless_asked(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        main(["stats", "data1"])
        capsys.readouterr()
        assert list(tmp_path.iterdir()) == []

    def test_figures_are_written_when_asked(self, tmp_path, capsys):
        main(["stats", "data1", "--save-figures", str(tmp_path)])
        capsys.readouterr()
        written = list(tmp_path.iterdir())
        assert written and all(p.stat().st_size > 0 for p in written)


class TestMarketCommand:
    def test_it_reports_a_summary_table(self, capsys):
        assert (
            main(["market", "baseline", "--sessions", "2", "--seconds", "120", "--periods", "2"])
            == 0
        )
        out = capsys.readouterr().out
        assert "scenario" in out and "equilibrium" in out

    def test_the_same_seed_gives_the_same_output(self, capsys):
        argv = [
            "market",
            "baseline",
            "--sessions",
            "2",
            "--seconds",
            "120",
            "--periods",
            "2",
            "--seed",
            "7",
        ]
        main(argv)
        first = capsys.readouterr().out
        main(argv)
        assert capsys.readouterr().out == first


class TestSentimentCommand:
    def test_it_analyses_a_named_text(self, capsys):
        assert main(["sentiment", "hack_crash_tweet"]) == 0
        assert "hack_crash_tweet" in capsys.readouterr().out

    def test_it_accepts_arbitrary_text(self, capsys):
        assert main(["sentiment", "This laptop is wonderful."]) == 0
        assert capsys.readouterr().out.strip()
