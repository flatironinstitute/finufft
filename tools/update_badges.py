#!/usr/bin/env python3
"""Publish self-hosted shields.io endpoint JSON for the CI status table.

shields.io's /github/check-runs badge only inspects GitHub's first page of
check runs (~30) and leans on shields.io's shared GitHub-API token pool, so on
a repo like finufft (60+ check runs per commit) the per-compiler badges
intermittently render "unknown status" or "Unable to select next GitHub token
from pool" for any job past page 1.

This script sidesteps both problems: it queries the latest *completed* run of
each CI workflow on the branch (paginating its jobs itself, so no 30-row cap)
and writes one shields "endpoint" JSON per table row. shields then renders our
public JSON from raw.githubusercontent without ever touching GitHub's API, so
the badges are reliable and need no third-party app access.

We read the latest *completed* run rather than the branch HEAD's check runs so
that a ``[no ci]`` commit (which starts no run) leaves the badges unchanged
instead of blanking them to "no status".

Run by .github/workflows/badges.yml; the JSON is force-pushed to the `badges`
branch and referenced from README.md via tools/gen_platform_table.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gen_platform_table import badge_slug, collect_rows

# GitHub run / job conclusion -> (badge message, badge colour).
CONCLUSION_BADGE = {
    "success": ("passing", "brightgreen"),
    "failure": ("failing", "red"),
    "timed_out": ("timed out", "red"),
    "startup_failure": ("startup failure", "red"),
    "action_required": ("action required", "yellow"),
    "cancelled": ("cancelled", "lightgrey"),
    "skipped": ("skipped", "lightgrey"),
    "neutral": ("neutral", "lightgrey"),
    "stale": ("stale", "lightgrey"),
}


def _gh_api(args: list[str]) -> str:
    return subprocess.run(
        ["gh", "api", *args], check=True, capture_output=True, text=True
    ).stdout


def _badge_message(status: str, conclusion: str) -> tuple[str, str]:
    if status and status != "completed":
        return ("running", "blue")
    return CONCLUSION_BADGE.get(conclusion, ("no status", "lightgrey"))


def latest_completed_run(repo: str, ref: str, workflow: str) -> tuple[str, str]:
    """(run_id, conclusion) of the most recent completed run of `workflow` on `ref`.

    Returns ("", "") if the workflow has no completed run on the branch.
    """
    out = _gh_api(
        [
            f"repos/{repo}/actions/workflows/{workflow}/runs"
            f"?branch={ref}&status=completed&per_page=1",
            "--jq",
            r'.workflow_runs[0] | [(.id | tostring), (.conclusion // "")] | @tsv',
        ]
    ).strip()
    run_id, conclusion = (out.split("\t") + ["", ""])[:2]
    return (run_id, conclusion)


def run_job_states(repo: str, run_id: str) -> dict[str, tuple[str, str]]:
    """Map each job name in a workflow run to (status, conclusion)."""
    out = _gh_api(
        [
            "--paginate",
            f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
            "--jq",
            r'.jobs[] | [.name, .status, (.conclusion // "")] | @tsv',
        ]
    )
    states: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        name, status, conclusion = (line.split("\t") + ["", ""])[:3]
        states.setdefault(name, (status, conclusion))
    return states


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="flatironinstitute/finufft")
    p.add_argument("--ref", default="master")
    p.add_argument(
        "--output", type=Path, required=True, help="directory for the JSON files"
    )
    args = p.parse_args()

    rows = collect_rows()

    # One latest-completed-run lookup per workflow; collect its per-job states
    # and its overall conclusion (the fallback for rows whose YAML job key is
    # not the rendered check name, i.e. the C++.yml make jobs).
    job_states: dict[str, tuple[str, str]] = {}
    workflow_conclusion: dict[str, str] = {}
    for workflow in sorted({r.workflow for r in rows}):
        run_id, conclusion = latest_completed_run(args.repo, args.ref, workflow)
        workflow_conclusion[workflow] = conclusion
        if run_id:
            job_states.update(run_job_states(args.repo, run_id))

    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.job_name in job_states:
            status, conclusion = job_states[row.job_name]
        else:
            status, conclusion = "completed", workflow_conclusion[row.workflow]
        message, color = _badge_message(status, conclusion)
        payload = {
            "schemaVersion": 1,
            "label": row.compiler,
            "message": message,
            "color": color,
        }
        (args.output / f"{badge_slug(row)}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    print(f"wrote {len(rows)} badge files to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
