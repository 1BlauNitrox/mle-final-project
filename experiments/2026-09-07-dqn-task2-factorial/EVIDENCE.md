# Issue #107 evidence reproduction

The registered experiment result was independently reproduced from the server
download. All 5,180 jobs and twenty final checkpoints passed verification;
the complete recomputed JSON equals `server-analysis/result.json`.

## Contents

The compressed archive retains campaign authorization/resource records, all six
resolved plans and status files, per-attempt metadata, normalized training and
evaluation episode rows, and checkpoint bytes referenced by the audit. Verbose
logs, replay collections outside required checkpoints, redundant input-agent
snapshots and framework-statistics dumps are excluded. `verification.json`
records SHA-256 and byte size for each retained file, execution and analysis
environments, verified final artifacts and failed attempts (none).

## Retrieval and verification

Release: `issue107-evidence-v1` in `1BlauNitrox/mle-final-project`.
Asset: `issue107-evidence-v1.tar.gz`. Its checksum and byte size are recorded
in `evidence-manifest.json`. The release preserves the original execution
revision, independently of later result documentation.

From a checkout containing this result PR, download the archive to an external
evidence directory and verify its SHA-256 before extraction:

```bash
gh release download issue107-evidence-v1 --repo 1BlauNitrox/mle-final-project \
  --pattern issue107-evidence-v1.tar.gz --dir /path/to/evidence
sha256sum /path/to/evidence/issue107-evidence-v1.tar.gz
tar -xzf /path/to/evidence/issue107-evidence-v1.tar.gz -C /path/to/evidence
git worktree add --detach /path/to/issue107-execution \
  69e2a931e4ebfd358cc0128049516b1a12adc2a8
```

Compare the printed archive hash against `evidence-manifest.json`. On Windows
use `Get-FileHash -Algorithm SHA256` instead of `sha256sum`.

Install the documented analysis dependencies in a separate environment, then
run the verification script from the clean execution checkout. The script
itself comes from the result checkout; imported analysis code comes from the
exact execution revision:

```bash
cd /path/to/issue107-execution
python /path/to/result-checkout/scripts/verify_issue107_download.py \
  --plan-root /path/to/evidence/run-plans \
  --output /path/to/reproduced-analysis \
  --expected-result /path/to/result-checkout/experiments/2026-09-07-dqn-task2-factorial/server-analysis/result.json
```

Expected output includes `jobs_verified: 5180`, `exact_server_result_match: true`
and `analysis_valid: true`. The output directory also contains the recomputed
tables and a new `verification.json`. Host-specific analysis environment fields
may differ; the complete result must match exactly. This command performs no
training or game evaluation.

## Execution versus analysis provenance

The original analyzer constructs registered-plan fingerprints using installed
packages and native path ordering. The portable wrapper verifies a clean exact
execution revision, reconstructs the Linux agent-file ordering, checks every
plan/job against a consistent hashed execution dependency inventory, and
records analysis dependencies separately. It does not replace source or
framework hashes with reported values, and the source, configuration, agent
bytes and parent hashes must match the registered plan. All statistical and
decision functions are the unchanged registered functions.

Local reproduction used Python 3.13.14 and NumPy 2.5.1; execution used Python
3.14.4 and NumPy 2.5.3. Despite that difference, every recomputed result value
matched. The directory-ordering adaptation addresses filename ordering, not
content changes. These implementation details are covered by integrity tests.

The original selected training checkpoint remains available in the archive at
`run-plans/issue107-cell-a-control/artifacts/r2/classic-crates/checkpoint.pt`.
Its SHA-256 is
`d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d`,
and its size is 2,382,903 bytes. It is an exploratory Task 3 predecessor;
this release does not freeze a submission candidate.

## Local data retention

The published archive contains the evidence required to reproduce this result,
including the selected checkpoint. After checking a downloaded release against
the manifest, the bulky local server-output copy is dispensable for these
claims. Keep the 36.9 MB archive as a convenient offline copy. Preserve the
server originals through non-author review, particularly verbose logs and
trajectories excluded from this compact archive: those may still help diagnose
learning behavior, although they are not needed to recompute the reported
statistics. This advice applies only to Issue #107 downloads, not other
experiments, repository worktrees, or uncommitted work.

The exporter verifies every finished archive member against the audit after
packaging, rejecting altered bytes, missing files, duplicates, unexpected
entries and links. This closes the interval between source-file auditing and
archive creation; no archive is reported successful without that final check.
