# 0008 Task 3 Opponent-Awareness Contract

## Cumulative successor

`DagobertDuckDQNTask3` is a self-contained successor of `DagobertDuckDQNTask2`.
Feature schema 4 has 39 inputs: all 26 Task 2 inputs, followed by the existing
13 public-opponent inputs. Indices 0-20 retain navigation and crate/bomb
features; indices 21-25 retain the five timed surviving-continuation flags.
The continuation implementation is vendored inside Task 3 and differentially
tested against Task 2. It imports no parent agent at evaluation time.

The parent checkpoint controls whether continuation features are active and
whether actions/Bellman targets use framework-legal masking. Missing environment
selectors preserve those modes; explicit mismatches fail in training and
evaluation, including fresh migrations. No mode is silently adopted as a new
default. The default fixture retains disabled escape and masking.

## Opponent suffix

| Index | Feature | Domain and definition |
| ---: | --- | --- |
| 26 | `opponent_visible` | `1` when at least one public opponent exists, else `0` |
| 27 | `opponent_dx` | Sign of nearest opponent x-coordinate minus own x-coordinate |
| 28 | `opponent_dy` | Sign of nearest opponent y-coordinate minus own y-coordinate |
| 29 | `opponent_distance_bin` | Manhattan distance to the nearest opponent: `0` absent, `1` for 1, `2` for 2--3, `3` for 4+ |
| 30 | `opponent_attack_opportunity` | `1` when a bomb at the current position would include any opponent in its wall-blocked blast footprint |
| 31 | `opponent_attack_escape_exists` | `1` only when an attack opportunity exists and a time-safe escape remains after that hypothetical bomb |
| 32 | `opponent_adjacent_up` | `1` when a public opponent occupies the tile one step up |
| 33 | `opponent_adjacent_right` | `1` when a public opponent occupies the tile one step right |
| 34 | `opponent_adjacent_down` | `1` when a public opponent occupies the tile one step down |
| 35 | `opponent_adjacent_left` | `1` when a public opponent occupies the tile one step left |
| 36 | `second_opponent_dx` | Sign of second-nearest opponent x-coordinate minus own x-coordinate; `0` when fewer than two opponents are present |
| 37 | `second_opponent_dy` | Sign of second-nearest opponent y-coordinate minus own y-coordinate; `0` when fewer than two opponents are present |
| 38 | `second_opponent_distance_bin` | Manhattan distance to the second-nearest opponent, same bins as index 24; `0` when fewer than two opponents are present |

Nearest and second-nearest opponents are ordered by Manhattan distance, then
x/y coordinates. Only public positions are used. Opponents remain obstacles
in movement, crate search and timed escape. All thirteen suffix values are
zero without opponents. Normalize indices 7, 9, 18, 19, 29 and 38 by dividing
by three; all other values retain their existing signed/binary encoding.

## Learning and attribution

The network is `39 -> 64 -> 64 -> 6`, with actions in order
`UP RIGHT DOWN LEFT WAIT BOMB`. Retain the parent's applicable hyperparameters,
masking behavior, Task 1/2 rewards and one-CPU-thread execution contract.
Only the native `KILLED_OPPONENT` event adds reward (+5 per occurrence).
`OPPONENT_ELIMINATED` is diagnostic and unrewarded because it may describe
another agent's kill. Terminal events are counted exactly once, including
simultaneous elimination and death. No opponent action oracle is introduced.

## Migration and artifacts

Migration accepts legacy 21-input parents and 26-input parents with escape
features off or on. Copy every available parent input column, hidden layer,
bias and all six outputs independently for online and target networks. Zero
all thirteen opponent columns; legacy parents additionally receive five zero
escape columns with the feature mode disabled. This preserves inherited
Q-values within `1e-5` float32 tolerance. New opponent columns can receive
training gradients. Schema-3/34-input Task 3 checkpoints are incompatible and
are rejected, rather than silently reinterpreted.

Start a new optimizer, empty replay, initial epsilon, fresh seed streams and
zero Task 3 episode/update counts. Parent replay/optimizer state is not loaded
into the new schema. The migration CLI requires the parent SHA-256 and rejects
source overwrite. The committed fixture is a reproducible migration of
`checkpoint-issue85-zero-suffix.pt` (SHA-256
`3edb2e7196030fcb52af6c7dc9ee69d9fc1259898ea674002fe06fbe93468015`), not a selected
Task 2 result. Its current checksum and size are in the Task 3 artifact manifest.

## Scientific boundary

Issue #124's mechanical selection will bind the next Task 3 campaign parent.
If Task 2 gates fail, label the continuation exploratory; it does not finish
Task 2. The verified historical #107 A/r2 parent can support an explicitly
separate exploratory baseline before that binding, never a silent substitution.
Issue #109 / PR #120 remains the single peaceful-opponent protocol. Its numeric
criteria, analyzer and explicit compute budget must be finalized before a
scientific run. The coin-collector experiment remains downstream of the
peaceful-stage decision. No result or final tournament readiness is claimed.
