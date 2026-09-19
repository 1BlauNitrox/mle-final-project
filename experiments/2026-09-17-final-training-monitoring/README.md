# Monitoring suite: 100 fresh classic worlds

This is not an experiment. We use this registration only to pin a world set, so we judge every milestone evaluated during a run on the same boards.

## Why it exists

Our first milestone evaluations used the 40 development worlds from the final training registration. They were too noisy to be useful. Score on 40 worlds carries about +/-0.9, so these evaluations did not detect a real +0.24 gain already visible in the training logs. This creates a risk of calling a run flat when it is not.

We therefore registered 100 fresh seeds, 970000001 to 970000100. We audited them against every integer in the tracked files and against the held-out suite. `scenario` is classic and the opponents are three `rule_based_agent`s. This matches the tournament line-up and the setting used for our primary endpoint.

## What uses it

* [`scripts/evaluate_milestones.py --registration`](../../scripts/evaluate_milestones.py) for rolling monitoring of the final training run, with results on `monitoring/final-training-evaluation`
* [`scripts/monitor_warm_lineup.py`](../../scripts/monitor_warm_lineup.py) for the continuation run, where we evaluate each new milestone against the agent the run started from

## What it is not for

These are development worlds. We play them repeatedly with many checkpoints, so we do not select anything based on them without introducing the winner's curse. The final training run's own data showed a split-half rank correlation of -0.22 between the two halves of these worlds. Ranking checkpoints on them therefore measures noise.

We keep the held-out suite `holdout-rule-based` unplayed until we have already chosen the artifact we will ship.
