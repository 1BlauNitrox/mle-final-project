# History-aware anti-loop pilot

Issue #221 tests whether causal visit counts together with a small conditional movement penalty reduce persistent loops while retaining the capabilities of the frozen fallback.

This is a bundled intervention. The experiment therefore measures the combined effect of the history features and reward shaping. It does not distinguish their individual contributions.

We ran the experiment on this PC until 14:00 Berlin on 20 September 2026 under a shared two CPU-hour limit covering mechanics, training, evaluation, and recovery.

## Setup

The executable experiment definition is stored in `config.json`.

We use two paired replicas. Both start from the fixed control-r1 milestone008000 fallback with SHA-256:

`a5dadff5cc9f04a614caa828d3a0f014cad56f00c3644bb791275da85b35a8a6`

The runtime is archived from commit:

`c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113`

Both arms preserve the online and target-network weights, Adam state, and replay buffer.

The experiment uses the same 56-input architecture inherited from #211. Twelve geometry inputs remain zero in both arms.

Historical replay contains no reconstructible visit history. We therefore zero-pad the new history columns for these transitions.

The control keeps the history suffix inactive. The treatment uses five visit-count features based on the preceding 16 observations.

## Anti-loop intervention

The treatment adds a training penalty of -0.1 for a successful directional move back into a persistent cycle.

The preceding 24 observations must:

* cover at most three positions
* contain no bombs or explosions
* show no change in the field
* show no change in visible coins
* show no change in score
* show no change in opponent count

We exclude transitions involving progress, invalid moves, BOMB, or WAIT.

The network still selects every action itself. We do not apply a hand-written action override.

## Training protocol

Both arms train on paired Classic worlds against three `rule_based_agent` opponents.

We use:

* learning rate 0.0001
* one update every four transitions
* one seeded random episode in every block of five episodes
* fixed checkpoints after 25, 100, 250, and 1,000 new updates
* a maximum of 200 episodes per arm

If an arm does not reach the registered update endpoint, we treat the run as incomplete.

Each early checkpoint, including the final one, must pass the registered fallback-retention gates before training continues. If a paired checkpoint fails, we stop that pair.

The exact pilot and final development suites, game counts, and seeds are stored in `config.json`. `seed-audit.json` contains the seed-collision audit.

Evaluation is greedy. Compared policies use paired worlds, opponent slots, and agent seeds.

## Metrics and decision rule

We record:

* native score
* kills
* survival
* self-kills
* collection
* invalid actions
* decision time
* loop exposure
* attack opportunities
* replay origins

The final exploratory screen requires at least a 25% reduction in eligible Classic loop rate compared with both the control and the frozen fallback.

The treatment must also pass the registered score, safety, collection, and action-correctness gates.

A run with no eligible loop windows does not pass the loop criterion.

We do not select intermediate checkpoints based on their results, and passing this screen does not automatically promote a checkpoint for submission.

## Reproduction

We recompute the generated analysis with:

```powershell
python -m scripts.analyze_antiloop_pilot --root <run-root> --output <new-analysis.json>
```

The final experiment record must include the completed results, our interpretation, and durable evidence locations after the run.
