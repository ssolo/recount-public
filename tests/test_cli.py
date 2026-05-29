import json
import subprocess
import sys
from pathlib import Path

from recount.io.countxml_writer import write_countxml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_countxml_fixture(tmp_path, tree, rates, profiles):
    path = tmp_path / "fixture.countxml.gz"
    write_countxml(
        path,
        tree,
        rates,
        [f"fam{i}" for i in range(profiles.shape[0])],
        profiles,
        session_id="cli-test",
        table_name="profiles.tsv",
    )
    return path


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "recount.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_cli_ll_accepts_min_copies_above_two(tmp_path, tree, rates, profiles):
    path = _write_countxml_fixture(tmp_path, tree, rates, profiles)

    result = _run_cli([
        "ll",
        str(path),
        "--table",
        "profiles.tsv",
        "--min-copies",
        "4",
        "--num-threads",
        "1",
    ])

    assert "# min_copies: 4" in result.stdout
    assert "LL_corrected_min4" in result.stdout
    assert "invalid choice" not in result.stderr


def test_cli_fit_stdout_is_clean_rate_table(tmp_path, tree, rates, profiles):
    path = _write_countxml_fixture(tmp_path, tree, rates, profiles[:2])

    result = _run_cli([
        "fit",
        str(path),
        "--table",
        "profiles.tsv",
        "--max-families",
        "2",
        "--max-iter",
        "1",
    ])

    first_line = result.stdout.splitlines()[0]
    assert first_line == "node\ttype\tname\tlength\tgain\tloss\tdup"
    assert "iter" not in result.stdout
    assert "# starting optimization" in result.stderr


def test_cli_events_accepts_countxml_jobs(tmp_path, tree, rates, profiles):
    path = _write_countxml_fixture(tmp_path, tree, rates, profiles)
    out_json = tmp_path / "events.json"

    result = _run_cli([
        "events",
        str(path),
        "--table",
        "profiles.tsv",
        "--max-families",
        "2",
        "--num-threads",
        "1",
        "--out-json",
        str(out_json),
    ])

    assert "gain_events" in result.stdout
    assert "loss_events" in result.stdout
    assert "# tree:" in result.stderr
    payload = json.loads(out_json.read_text())
    assert "families_present" in payload
    assert len(payload["families_present"]) == tree.num_nodes
