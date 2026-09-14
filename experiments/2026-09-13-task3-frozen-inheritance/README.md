# Frozen inherited weights: Task 3 experiment (#173)



We compared ordinary DQN fine-tuning with training only the 832 opponent-input
weights. Both started from the same Task 3 initialization. The experiment
kept all inherited online weights fixed. Five paired replicas trained for 100
episodes each, all ten final checkpoints and the unchanged reference were
evaluated on 40 shared development worlds in four suites, with exact repeats.

|Metric|Reference|Full-network control|Frozen-weight treatment|
|-|-:|-:|-:|
|Peaceful eliminations/game|0.200|0.185|0.170|
|Peaceful mean score|4.825|4.315|2.820|
|Peaceful strict first-place rate|75.0%|76.0%|61.5%|
|Peaceful self-kill rate|30.0%|22.0%|25.5%|
|Coin-heaven collection|60.05%|44.63%|60.05%|
|Loot-crate collection|17.00%|15.47%|17.00%|
|Solo classic collection|43.06%|32.06%|43.06%|



The registered experiment failed.
The hunting difference was -0.015
eliminations/game, below the required +0.100.
Only two of five treatment replicas matched or exceeded the reference's hunting.
We didnt select or promote any checkpoint.

The useful finding is preservation: every frozen tensor stayed identical and
all three earlier-task collection/self-kill retention gates passed. This retains
the parent's limitations. It does not establish cumulative Task 2 completion.

Repeatability passed 1,759/1,760 pairs. The one mismatch was also the only latency
outlier: a 1.186-second decision, versus 17.58 ms p95, followed by one skipped
action. That episode's score, collection and survival were unchanged, but the
strict repeat/invariance and maximum-latency gates still fail.
A separate replay matched treatment/reference actions and behavior (maximum decision times34.15
and19.06ms). It does not replace the original observations. Evaluation also
contained 26 invalid Actions, the zero-invalid gate fails.

The next controlled question is whether a stronger attributable-opponent reward
can improve hunting while retaining the frozen inherited policy. Keep the fresh
initialization, all other rewards and the earlier-task gates fixed; compare
KILLED\_OPPONENT +5 with +20. This is a hypothesis, not an adopted improvement.

See [config.json](config.json) for the registered protocol and
[results/analysis.json](results/analysis.json) for every gate and confidence
interval. [results/observations.csv](results/observations.csv) retains all
per-episode metrics.
