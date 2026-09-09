# Task 3 escape-preserving migration preparation

Refs #125, #108 and #109. Draft design only; no agent code, checkpoint or
training behavior is changed by this PR.

An escape-enabled Task 2 parent has 26 inputs. The existing Task 3 successor
has 34 inputs: a 21-input Task 2 prefix and thirteen opponent inputs. Its
control-only migration deliberately rejects an active-escape parent. The
proposed successor retains all 26 parent inputs in order and appends the same
thirteen opponent inputs, producing 39 inputs.

## Required implementation and tests

1. Keep the first 26 normalized values byte-for-byte equivalent to the parent,
   including five multi-step continuation inputs, with no imports from another
   agent at evaluation time.
2. Append the existing opponent suffix in its documented order. Update schema,
   persistence validation, network configuration, agent card and contract.
3. Copy all 26 input columns, hidden layers and six output rows. Zero all
   thirteen new columns. Test online and target Q-value preservation with a
   zero opponent suffix and random/nonzero escape inputs.
4. Use a fresh optimizer/replay for the new experiment unless its protocol
   explicitly specifies continuation; record this migration choice. Do not
   silently load incompatible replay arrays.
5. Carry the selected mask mode into behavior and Bellman targets; test it.
6. Require the parent SHA-256, refuse overwriting it, and document supported
   legacy schemas. Never drop active features to fit an older model.
7. Run Task 1/2 regression, package-only clean-framework and official Docker
   checks. Keep actual tournament resource validation separate from CI smoke.

## Dependencies and execution boundary

Implementation can proceed while #124 trains. Final binding waits for its
registered selection and retained artifact; no best-seed choice. Integrate with
existing PR #120, not a competing peaceful-opponent experiment. If Task 2 still
fails, record the parent as exploratory. #109 still needs numeric decision
rules, its analyzer, a shared resource monitor and compute authorization.

This PR intentionally has no launch command. Its remaining implementation is
listed above; it does not claim a working 39-input successor.
