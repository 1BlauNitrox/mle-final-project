# Task 3 preparation and launch dependencies

This is preparation only. No Task 3 training is authorized or started.
PR #115 is merged; PR #120 now targets main. Issue #107's downloaded server
analysis reports control A/r2 as the mechanical selection and Task 2 incomplete.
Independent evidence verification is pending in PR #122. Preserve that
selection until the verified record is reviewed; do not substitute B because
its measured survival is stronger.

## Parent binding after Issue #107 verification

The reported parent hash is
`d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d`
(2,382,903 bytes). It is a 26-input control checkpoint with escape continuations
off. The Task 3 schema has 21 inherited inputs and 13 opponent inputs. Migration
copies the first 21 parent columns and all later layers, discards only the five
always-zero continuation inputs, and zeroes all opponent columns. A parent with
active escape continuations is rejected because discarding it would change the
policy. Supporting that parent requires a separately reviewed schema decision.

After verifying the selected artifact, use an explicit output path:

```bash
python scripts/migrate_task3_dqn_successor.py \
  --parent /path/to/verified/issue107-control-r2-checkpoint.pt \
  --parent-sha256 d571b199bc00a0419d16706cc5f7844364a184384058268b2cc8de5b79bb479d \
  --output /path/to/new/task3-migration.pt
```

The tool verifies the parent hash and refuses identical input/output paths.
Optimizer, replay, exploration and RNG state start fresh. Bind the verified
parent and migration into both plans and manifests, including source revision,
checksums, feature/reward schema and inherited failed Task 2 gates. Dry-run
success alone does not verify checkpoint compatibility or authorize training.

## Matched comparison

The frozen reference now receives the same eight evaluation suites as each
candidate replica: peaceful-opponent classic, and opponent-free classic,
coin-heaven and loot-crate, each with primary/repeat passes. This supplies
paired retention measurements against the actual selected parent. There are
405 candidate jobs and 80 reference jobs, with 50,000 training episodes total.
Existing retention seeds are explicitly reused development seeds, not unseen
confirmation evidence. Keep all confirmation/final populations unopened.

## Remaining scientific launch gates

1. Complete #107 evidence verification and bind its selected parent.
2. Assign the owner and non-author reviewer.
3. Agree numerical elimination, paired score/first-place, self-kill, retention
   and fallback criteria before training, as required by #109. They remain
   unset; the preparation does not invent team approval.
4. Implement and test the result analyzer against the approved criteria,
   including observed-agent filtering, deterministic repeats, paired
   uncertainty, immutable provenance and mechanical checkpoint selection.
5. Validate aggregate CPU/wall/RAM enforcement and explicitly authorize the
   actual server budget. The proposed 24 CPU-hour/15 wall-hour/8 GiB ceiling
   is not an authorization; independent generic runner commands do not enforce
   a shared campaign ceiling.
6. Obtain review and green CI on the final immutable execution revision.

The coin-collector plan remains a gated template. It currently points at the
provisional migration, not a peaceful-stage selected checkpoint. Rebind it only
after a prospective peaceful-to-coin-collector decision, with a paired frozen
reference and agreed criteria. Do not execute it merely because its YAML parses.

This study must be labeled exploratory if the verified parent fails Task 2.
It does not close Task 2 or authorize more Task 2 optimization. No checkpoint
has been regenerated during this preparation.
