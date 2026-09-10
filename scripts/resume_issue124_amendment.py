"""Resume the immutable Issue 124 campaign under a recorded owner budget amendment.

Run this script from outside the execution worktree. The original authorization,
source SHA, plans, seeds, artifacts and job metadata are retained unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_amendment(amendment, authorization):
    old = authorization["limits"]
    new = amendment["new_limits"]
    if (
        amendment["original_limits"] != old
        or amendment["execution_commit"] != authorization["reviewed_commit"]
        or amendment["authorized_by"] != authorization["authorized_by"]
        or new["memory_bytes"] != old["memory_bytes"]
        or set(new) != set(old)
        or new["wall_seconds"] <= old["wall_seconds"]
        or new["cpu_seconds"] <= old["cpu_seconds"]
        or new["wall_seconds"] != 86400
        or new["cpu_seconds"] != 345600
        or amendment["workers"] != 4
    ):
        raise ValueError("Invalid Issue 124 budget-only amendment")


def resume_monitor_type(original_monitor):
    class ResumeMonitor(original_monitor):
        def __init__(self, **kwargs):
            # Preserve the original authorization; normalize only this argument
            # to the pinned monitor's persisted UTC spelling.
            kwargs["authorized_at"] = kwargs["authorized_at"].replace("+00:00", "Z")
            super().__init__(**kwargs)

    return ResumeMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    amendment_path = root / "budget-amendment.json"
    amendment = json.loads(amendment_path.read_text())
    authorization_path = root / "authorization.json"
    authorization = json.loads(authorization_path.read_text())
    validate_amendment(amendment, authorization)
    if digest(authorization_path) != amendment["original_authorization_sha256"]:
        raise ValueError("Original authorization changed")
    if digest(__file__) != amendment["resume_script_sha256"]:
        raise ValueError("Amendment runner changed")
    sys.path.insert(0, str(args.execution_root.resolve()))
    from training import run_issue124_campaign as campaign

    if campaign.ROOT.resolve() != args.execution_root.resolve():
        raise ValueError("Wrong execution module root")
    if campaign.git("rev-parse", "HEAD") != amendment["execution_commit"]:
        raise ValueError("Original execution source changed")
    campaign.validate_protocol()
    campaign.LIMITS = campaign.CampaignLimits(**amendment["new_limits"])
    campaign.LocalMonitor = resume_monitor_type(campaign.LocalMonitor)
    # Import after selecting effective limits: analysis uses this same amendment.
    from training import analyze_issue124_campaign as analysis

    original_analyze = analysis.analyze

    def analyze_with_amendment(output):
        result = original_analyze(output)
        result["budget_amendment"] = amendment
        campaign.write_json(root / "analysis/result.json", result)
        manifest = campaign.read_json(root / "analysis/evidence-files.json")
        for path in [amendment_path, Path(__file__).resolve()]:
            manifest[path.relative_to(root).as_posix()] = {
                "sha256": digest(path),
                "size_bytes": path.stat().st_size,
            }
        for relative in amendment["preserved_records"]:
            path = root / relative
            manifest[relative] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        campaign.write_json(root / "analysis/evidence-files.json", manifest)
        return result

    analysis.analyze = analyze_with_amendment
    if args.validate_only:
        print(
            json.dumps(
                {
                    "execution_commit": amendment["execution_commit"],
                    "effective_limits": vars(campaign.LIMITS),
                    "training_started": False,
                }
            )
        )
    elif args.analyze_only:
        result = analyze_with_amendment(root)
        print(
            json.dumps(
                {"analysis_valid": result["analysis_valid"], "selection": result["selection"]}
            )
        )
    else:
        campaign.execute(root, amendment["execution_commit"], resume=True, owner_override=True)


if __name__ == "__main__":
    main()
