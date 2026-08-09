# 0001 Project Requirements

This document translates the supplied 2026 final-project handout into an
actionable repository checklist. The handout remains authoritative if this
summary differs from it.

## Goal and grading emphasis

The team must train a machine-learning agent to play Bomberman. Tournament
performance contributes to the grade, but the systematic scientific design,
optimization, testing, and reporting process carries substantially more weight.

Required approach:

- implement at least two different learned models;
- include at least one model focused on lecture techniques;
- work collaboratively across models rather than assigning one isolated model
  to each person;
- divide the problem into manageable subgoals;
- compare variants against earlier versions and supplied baseline agents;
- define meaningful performance metrics;
- use experiments to motivate each subsequent modification;
- describe successful and abandoned approaches.

A rule-based solution without machine learning is not accepted.

## Deadlines and deliverables

### Optional compatibility test

- Deadline: 17 September 2026 at 21:00.
- Upload an agent zip to the MaMPF submission test.
- The course team installs libraries from `requirements.txt`, locates the first
  directory containing `callbacks.py`, and runs one non-training game against
  three `random_agent` instances.

### Agent-code submission

- Deadline: 21 September 2026 at 21:00.
- Submit `final-project-agent-code.zip` through MaMPF.
- The archive contains the directory of the best-performing agent, including
  every trained parameter and runtime dependency file it needs.
- Join the team submission in MaMPF before the deadline and use the same real
  name as in MUESLI.

### Report submission

- Deadline: 28 September 2026 at 21:00.
- Submit a PDF of about 4,000 words per team member, and not much more.
- Identify the main responsible author after each heading.
- Include the URL of this public repository.
- Do not commit the report PDF to this repository.
- Do not use the university logo for legal reasons.

The expected report structure is:

1. Introduction
2. Background
3. Project planning
4. Methods
5. Training
6. Experiments and Results
7. Conclusion

Experiments and Results is the most important report section. It must show
systematically whether design and implementation changes improved performance
and justify the selected final agent.

## Runtime constraints

- Official games run neural networks on CPU, even if training used a GPU.
- The agent gets exclusive access to one thread of an AMD Ryzen 5 2600 and up to
  8 GB RAM.
- `act()` has a 0.5-second time limit per step. Excess time reduces the next
  step's budget.
- An episode ends after 400 steps.
- Multiprocessing is forbidden in the final evaluation agent.
- Multiprocessing and other acceleration methods are allowed during training.
- The final agent is inserted into the original course framework. Changes to
  framework files are absent during official games.
- Agent code must therefore use relative paths and be self-contained.
- Free software libraries from official repositories such as pip or conda are
  allowed.
- Additional libraries must be declared in the repository, at the beginning of
  the report, and in an agent submission's `requirements.txt` when needed.

## Required framework interface

Every learned agent lives in `agent_code/<agent_name>/`.

Always loaded from `callbacks.py`:

- `setup(self)`
- `act(self, game_state)`

Loaded from `train.py` in training mode:

- `setup_training(self)`
- `game_events_occurred(self, old_game_state, self_action, new_game_state, events)`
- `end_of_round(self, last_game_state, last_action, events)`

All model files and trained parameters needed during evaluation must be stored
in the same agent directory.

## Progressive learning tasks

1. **Navigation:** Collect visible coins quickly on a board without crates or
   opponents.
2. **Bombing and survival:** Destroy crates, reveal coins, collect them, and
   learn to escape own bombs.
3. **Hunting:** Defeat `peaceful_agent`, then `coin_collector_agent`.
4. **Competition:** Compete for score against strong agents and own variants.
   Beating `rule_based_agent` is an important tournament-readiness target.

Example training commands:

```bash
# Task 1
python main.py play --no-gui --agents my_agent --train 1 --scenario coin-heaven

# Task 2
python main.py play --no-gui --agents my_agent --train 1 --scenario classic

# Task 3
python main.py play --no-gui --agents my_agent peaceful_agent \
  coin_collector_agent --train 1 --scenario classic

# Task 4
python main.py play --no-gui --agents my_agent rule_based_agent \
  --train 1 --scenario classic
```

## Recommended investigation areas

The handout recommends careful experiments around:

- feature engineering for situational awareness, pathfinding, and survival;
- deep learning, with enough time reserved for convergence and tuning;
- balanced auxiliary reward shaping and custom events;
- curriculum learning through custom scenarios or easier environments;
- hyperparameter optimization;
- rotational and mirror symmetries for data augmentation or state equivalence;
- self-play with enough action randomness to avoid poor local optima.

Feature engineering must not encode a deterministic rule-based best action in
place of learning.

## Game facts relevant to evaluation

- Actions are `UP`, `RIGHT`, `DOWN`, `LEFT`, `BOMB`, and `WAIT`.
- Coins score one point; eliminating an opponent scores five.
- Bombs detonate after four steps and reach three tiles in each cardinal
  direction unless blocked by a stone wall.
- Crates, bombs, walls, and agents block movement.
- Board randomness requires evaluation over multiple episodes and seeds.

## AI-use regulation

AI may support brainstorming, coding, and report drafting. The code and report
must make clear that the team performed the main work. Team members must
critically review, verify, and rewrite AI drafts in their own style. See
[`0006-ai-usage.md`](0006-ai-usage.md).

## Final checklist

- [ ] At least two learned models exist and are documented.
- [ ] At least one model uses lecture techniques.
- [ ] All important variants have controlled comparison results.
- [ ] The selected agent beats relevant baselines consistently.
- [ ] Evaluation respects CPU, memory, time, and multiprocessing constraints.
- [ ] The final agent is self-contained and uses relative paths.
- [ ] Dependencies are declared.
- [ ] The official Docker compatibility test passes.
- [ ] The submission-test feedback has been addressed.
- [ ] The best agent directory is packaged as the required zip.
- [ ] The public repository includes every developed model.
- [ ] The report is not present in the repository.
- [ ] The report identifies authors per section and links this repository.
