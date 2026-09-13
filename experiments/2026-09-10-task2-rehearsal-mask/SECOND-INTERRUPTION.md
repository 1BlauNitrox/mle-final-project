# Second memory interruption: September 11

At the 08:10 Berlin inspection the campaign was stopped, not completed.
The recorded stop occurred at 00:56:43 Berlin, again because system-wide
available RAM fell below 1 GiB. No campaign workers remain active.

A, B and C each completed 100/100 training blocks. D completed 95/100;
its remaining five blocks must finish before the serial evaluation barrier.
Thus 395/400 blocks (197,500/200,000 planned training episodes) are retained,
excluding partial failed attempts. All 5,120 evaluation episodes remain pending.

The September 10 recovery succeeded operationally and added ten complete
blocks, but did not prevent another system-memory interruption. Recorded
campaign peak memory remains 1.26 GiB, below the 8 GiB campaign ceiling.
Cumulative CPU use is 48.91 hours; the wall origin has not been reset.
The available evidence does not identify the process causing system pressure.

The registered amendment-aware `--analyze-only` command was retried on
September 11 and rejected the breached resource record. There is no valid
scientific selection, treatment comparison or Task 2 completion claim.
No agent optimization has been adopted from these incomplete runs.

`second-interruption-evidence.json` preserves the stopped status/resource
records, recovery amendment and per-arm counts with source hashes and remaining
training-job records. This supplements, rather than replaces, the first
interruption snapshot. It is not scientific performance evidence.

At inspection, available RAM was about 3.37 GiB, below the 4 GiB startup floor.
The existing wall deadline is September 11 08:59:45 Berlin, leaving roughly
50 minutes at inspection. Completion of five training blocks plus all evaluation
and analysis within that window is not established. Recovery needs stable free
RAM and a documented budget amendment if work extends beyond that deadline.
Do not delete outputs, reset accounting or silently change the registered
comparison to completed arms only. Keep Task 2 open; Task 3 preparation can
continue with its explicitly exploratory historical baseline.
