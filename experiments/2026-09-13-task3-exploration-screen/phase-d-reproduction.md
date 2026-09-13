# Reproduce the Issue 168 learning-pilot result

The result package is prepared locally. **The release has not been uploaded;
the retrieval command becomes usable after publication consent and upload.**
The local archive has been imported into a new directory and reanalyzed; its
result exactly matches `phase-d-analysis.json`. No training is needed.

## Evidence

- `phase-d-observations.json.gz`: lossless compact JSON, schema version 1,
  mapping training/evaluation job names to all 300/280 recorded episode rows.
  Rows retain seeds, native metrics, timing, actual exploration mode, weight
  hashes and learning counters. Exact repeats remain separate observations.
- `phase-d-summary.csv`: all seven evaluated artifacts across four suites,
  using primary episodes only. Repeats are integrity checks, not extra samples.
- `phase-d-analysis.json`: complete registered gates, per-model means and
  paired crossed replica/world bootstrap intervals (10,000 draws, seed 1684000).
- `phase-d-evidence-manifest.json`: checksums, sizes, exact source binding,
  all final checkpoint hashes, completed jobs and stage accounting.
- External archive `issue168-pilot-evidence.tar.gz`: 48,744,548 bytes,
  SHA-256 `d9f1a1e14ebb0734ba3cfddafa58e34aabd07393f962394f3d9625a3fe5b4071`.
  It retains full inputs, runtime source, final checkpoints and native outputs
  needed for the analyzer's tensor/source/integrity checks. No checkpoint was selected.

The experiment executed tool revision
`dcff79fefe812becbe76e7bdf2d1718a9eda4cea` with separately archived scientific
runtime `c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`.
The unchanged reference hash is
`99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113`;
the shared epsilon-0.2 training initialization hash is
`8076ea7ddcf2c934a9a9ef21d3dc87833ededf860cf8fd8bf486d1e222c2ccf2`.
Both derive from provisional #91 A/r3, not a selected #168 replica.
The archived runtime has 39 inputs; this branch's unmodified agent fixture has
34 inputs. Reproduction must use the archive through the bound pilot tool.

## Full verification in PowerShell

Use Python 3.13 with the recorded dependency versions in the manifest. Run
from this PR's checkout with unchanged pilot tools, or a separate clean checkout
of `dcff79fefe812becbe76e7bdf2d1718a9eda4cea`. Do not switch/reset a worktree
containing user edits or an active campaign. Use a new verification directory.

After the release has been published:

```powershell
$project = '1BlauNitrox/mle-final-project'
$tag = 'issue168-pilot-evidence-v1'
$archive = 'issue168-pilot-evidence.tar.gz'
$download = Join-Path $PWD 'training_outputs/issue168-download'
New-Item -ItemType Directory $download | Out-Null
gh release download $tag --repo $project --pattern $archive --dir $download
if ($LASTEXITCODE -ne 0) { throw 'Evidence download failed' }
$archive = Join-Path $download $archive
```

For an existing local copy, set `$archive` to that path and `$download` to a
new output directory instead. With `$py` pointing to the Python executable:

```powershell
$hash = 'd9f1a1e14ebb0734ba3cfddafa58e34aabd07393f962394f3d9625a3fe5b4071'
if ((Get-FileHash $archive).Hash.ToLowerInvariant() -ne $hash) {
    throw 'Archive checksum mismatch'
}
if ((Get-Item $archive).Length -ne 48744548) { throw 'Wrong archive size' }
$root = Join-Path $PWD 'training_outputs/issue168-reproduce'
$argsImport = @('import', '--root', $root, '--archive', $archive)
$argsImport += @('--sha256', $hash)
& $py -m scripts.pilot_task3_episode_exploration @argsImport
if ($LASTEXITCODE -ne 0) { throw 'Evidence import failed' }
$out = Join-Path $download 'recomputed-analysis.json'
& $py -m scripts.pilot_task3_episode_exploration analyze --root $root --output $out
if ($LASTEXITCODE -ne 0) { throw 'Evidence analysis failed' }
$record = 'experiments/2026-09-13-task3-exploration-screen'
$expected = Join-Path $record 'phase-d-analysis.json'
$check = 'import json,sys; from pathlib import Path; '
$check += 'assert json.loads(Path(sys.argv[1]).read_text()) == '
$check += 'json.loads(Path(sys.argv[2]).read_text())'
& $py -c $check $expected $out
if ($LASTEXITCODE -ne 0) { throw 'Analysis differs' }
```

The archive importer rejects existing roots and mismatched fingerprints.
Evaluation must be complete; its exact seed matrix, repeated behavior and
immutable model weights are checked. Training must use the registered paired
initialization and exploration schedule, and all six checkpoints must show
actual optimizer updates. No failed scientific seed was excluded.

All six training jobs and all 28 evaluation jobs completed in their first
attempt. Training used 1,162.50 CPU / 1,180.23 wall seconds; evaluation used
703.92 CPU / 719.74 wall seconds. Both stayed within their original ceilings.
The same PC performed both arms and all evaluations. Detailed environment and
resource records remain in the manifest, not in the scientific README.

Phase A's retained technical export failure and phase C's two reproduced
simultaneous-movement collisions remain in their original compact archives.
They are not omitted from those diagnostics and are not phase-D retries.
