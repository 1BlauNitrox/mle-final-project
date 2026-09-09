# 0008 Task 3 Opponent-Awareness Contract

## Decision

`DagobertDuckDQNTask3` is a separately named DQN successor of
`DagobertDuckDQNTask2`. It keeps the Task 2 six-action order, hidden layers,
normalization, rewards, action-masking default, and training lifecycle. Task 3
adds only descriptive opponent context and the native elimination reward.

The implementation uses the corrected Task 2 migration checkpoint as a
provisional parent while issue #107 selects the development artifact. The
selected Task 2 source and artifact hashes must be rebound in the agent
manifest after #107; no Task 3 result is used for that selection.

## Feature schema

Feature schema version 3 has 34 inputs. Indices 0--20 are copied unchanged
from Task 2. Indices 21--33 are appended in this order:

| Index | Feature | Domain and definition |
| ---: | --- | --- |
| 21 | `opponent_visible` | `1` when at least one public opponent exists, else `0` |
| 22 | `opponent_dx` | Sign of nearest opponent x-coordinate minus own x-coordinate |
| 23 | `opponent_dy` | Sign of nearest opponent y-coordinate minus own y-coordinate |
| 24 | `opponent_distance_bin` | Manhattan distance to the nearest opponent: `0` absent, `1` for 1, `2` for 2--3, `3` for 4+ |
| 25 | `opponent_attack_opportunity` | `1` when a bomb at the current position would include any opponent in its wall-blocked blast footprint |
| 26 | `opponent_attack_escape_exists` | `1` only when an attack opportunity exists and a time-safe escape remains after that hypothetical bomb |
| 27 | `opponent_adjacent_up` | `1` when a public opponent occupies the tile one step up |
| 28 | `opponent_adjacent_right` | `1` when a public opponent occupies the tile one step right |
| 29 | `opponent_adjacent_down` | `1` when a public opponent occupies the tile one step down |
| 30 | `opponent_adjacent_left` | `1` when a public opponent occupies the tile one step left |
| 31 | `second_opponent_dx` | Sign of second-nearest opponent x-coordinate minus own x-coordinate; `0` when fewer than two opponents are present |
| 32 | `second_opponent_dy` | Sign of second-nearest opponent y-coordinate minus own y-coordinate; `0` when fewer than two opponents are present |
| 33 | `second_opponent_distance_bin` | Manhattan distance to the second-nearest opponent, same bins as index 24; `0` when fewer than two opponents are present |

The four per-direction occupancy flags (27--30) mirror `free_directions`'
own per-direction ordering, so opponents count as obstacles in each specific
direction rather than as one aggregate count -- the original eight-value
suffix collapsed this into a single `adjacent_opponent_count_bin` and a
single `opponent_blast_count_bin`, which this schema replaces with the
per-direction flags and the second-nearest descriptors (31--33) so that
`classic`'s up to three simultaneous opponents are not collapsed into a
nearest-only signal. Opponents are obstacles in immediate movement,
crate-target BFS, and every non-wait step of the escape search. Both the
nearest and second-nearest tie-breaks are Manhattan distance followed by x
and y coordinate, so the representation is deterministic and does not
depend on agent ordering. When no opponent is present, all thirteen
appended raw values are zero.

Only the two distance bins are divided during normalization: indices 24 and
33 are each divided by 3. Signed direction values remain in `[-1, 1]`; the
binary values (including the four per-direction occupancy flags) are
already normalized.

## Elimination signal and attribution

All inherited Task 1/Task 2 reward values remain unchanged. Task 3 adds:

| Event | Reward | Attribution |
| --- | ---: | --- |
| `KILLED_OPPONENT` | `+5.0` | One native framework event per opponent killed by this agent's bomb |

`KILLED_OPPONENT` is the only new reward-bearing event. `OPPONENT_ELIMINATED`
is retained as a diagnostic event but has reward `0.0`, because it is emitted
to surviving agents when another agent gets the kill and is not attributable
to the learner. Existing custom Task 2 bomb-usefulness events remain
unchanged. No new action-oracle or opponent-action inference is introduced.

The event list is treated as an event multiset: each occurrence contributes
once. Terminal transitions use the complete terminal event list, including a
simultaneous elimination and self-death, and the pending-transition protocol
ensures that transition is recorded exactly once. A surviving final
transition is not updated once as ordinary and again as terminal.

## Migration and persistence

A 26-input Task 2 control parent is also supported when its persisted
`escape_continuation_features` flag is false: the last five inputs are always
zero, so their columns may be removed while preserving all six Q-values.
Parents with active continuation features are rejected; preserving those
requires a separately reviewed Task 3 schema. The migration CLI requires a
matching parent SHA-256 before loading and supports an explicit output path.

The provisional parent is the corrected Task 2 artifact
`checkpoint-issue85-zero-suffix.pt`, SHA-256
`3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015`. The
provisional parent source is current-main commit
`933a8fe11440e0f7645254390928da6af5dad46d`.

Migration copies all 21 inherited input columns, hidden layers, and all six
output rows and biases. The thirteen new input columns are zero-initialized,
so the initial Task 3 Q-values and greedy behavior equal Task 2 for every
state. The optimizer, replay buffer, epsilon schedule, and RNG streams are
reset as a new Task 3 training state. The checkpoint records feature schema 3,
the complete reward mapping, action order, and the Task 3 configuration;
loading an incompatible schema or reward mapping fails clearly.

This is a protocol and implementation decision, not a performance result.
Scientific training and selection remain downstream of issue #107.
