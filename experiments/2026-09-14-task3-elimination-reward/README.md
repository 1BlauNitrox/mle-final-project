# Stronger elimination reward with frozen inheritance (#175)



We tested whether increasing the reward for an attributable opponent elimination
from +5 to +20 improves hunting. Both arms froze inherited online weights and
trained only the 832 opponent-input weights of the same 39-input DQN. All other
rewards, initialization, exploration and optimizer settings were matched.
Five paired replicas trained for 100 episodes per arm. The ten final checkpoints
and unchanged reference were evaluated on 40 shared development worlds in each
of four suites, with two repeats: 1,000 training and 3,520 evaluation episodes.



|Peaceful-opponent metric|Unchanged reference|+5 control|+20 treatment|
|-|-:|-:|-:|
|Eliminations/game|0.250|0.080|0.090|
|Mean score|5.025|1.935|2.505|
|Strict first-place rate|92.5%|56.0%|67.5%|
|Self-kill rate|30.0%|18.0%|19.0%|
|Survival rate|70.0%|82.0%|81.0%|



The registered experiment failed.
The treatment-control elimination difference was +0.010/game (95% CI \[-0.080, +0.105]). This misses
both the required +0.100 improvement and positive confidence-interval lower bound.
Treatment also fell below the unchanged reference by 0.160 eliminations/game
(95% CI \[-0.295, -0.035]). None of its five replicas matched the reference.
The score and first-place point estimates improved over control, but their
intervals also include zero. They do not establish an improvement.



The useful result is exact earlier-task preservation. Both trained arms matched
the reference's opponent-free actions and behavior in all 2,400 comparisons.
Collection fractions were 53.60% on coin-heaven, 16.80% on loot-crate and 31.11%
on solo classic. All collection and self-kill retention gates passed. This
preserves the parent's limitations; cumulative Task 2 capability remains incomplete.

All 1,760 repeat pairs matched. Latency passed: median 5.40 ms, p95 20.14 ms and
maximum 36.37 ms. Frozen-weight, genuine-update and resource checks passed.
Eight invalid actions remained across both repeats (reference 2, control 0,
treatment 6), so the zero-invalid gate failed. All 54 scientific jobs completed
without failed attempts. Independent archive analysis reproduced every result exactly.



No checkpoint is selected or promoted, and +20 is not adopted as the default.
This experiment provides no evidence that simply increasing kill reward solves
hunting. Before another long run, the proposed follow-up is a short matched-state
diagnosis of how learned opponent inputs change movement and bombing decisions
relative to the unchanged reference, plus an inspection of the retained invalid
actions. Any intervention and new training budget must be registered separately.
These development seeds must not become a confirmation set. Results from #173
use different worlds and are not a matched comparison with this experiment.

[config.json](config.json) contains the prospective protocol.
[results/analysis.json](results/analysis.json) contains all gates and intervals;
[results/observations.csv](results/observations.csv) retains every evaluation row.
[results/verification.json](results/verification.json) records source, parent,
checkpoint and execution verification. The compact archive is verified locally;
[results/evidence.json](results/evidence.json) records its hashes and reproduction
commands.
