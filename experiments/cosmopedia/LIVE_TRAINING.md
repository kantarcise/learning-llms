# Live loss, stop and resume

The trainer writes a live loss curve while it runs, keeps the curve in resumable checkpoints, and exports `loss.svg` when it stops. Resuming restores the global update count, training/validation measurements and unfinished averaging windows, so the graph continues from the checkpoint rather than restarting at zero.

The dashboard uses Python's standard library and browser SVG. There are no new dependencies or hosted services. It binds only to `127.0.0.1`.

## Start a session

Run the dashboard in one terminal:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/live.py \
  --session experiments/cosmopedia/artifacts/my-session --port 8765
```

Open [the local dashboard](http://127.0.0.1:8765). Start training in another terminal, choosing a fresh output directory inside the same session folder:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --device cuda --session experiments/cosmopedia/artifacts/my-session \
  --output experiments/cosmopedia/artifacts/my-session/part-01 \
  --steps 100000 --eval-every 1000 --checkpoint-every 5000 --log-every 100 \
  --max-seconds 7200
```

This example uses the existing 1M-token data by default. For the new 10M-token experiment below, additionally pass `--data experiments/cosmopedia/artifacts/data-10m --validation-windows 300` on every invocation.

`--log-every 100` adds a training-loss point every 100 global updates, weighted by non-padding targets. Validation is measured separately every 1,000 updates. The dashboard refreshes every two seconds and labels the global update, historical best validation result, last saved update and resume starting point. A hollow training point shows an unfinished averaging window. The browser can download the current graph while training continues.

This is training with replacement: update counts are not epochs. Recent training-window and fixed-window validation losses have different measurement procedures. A smaller training loss does not by itself establish better generation or generalization.

## Stop safely

Click **Stop & save checkpoint**. Alternatively, send SIGINT/SIGTERM to the trainer or create a `STOP` file in its current output directory:

```bash
touch experiments/cosmopedia/artifacts/my-session/part-01/STOP
```

A request does not interrupt an optimizer update or checkpoint serialization. The trainer finishes the current work, saves a resumable latest checkpoint, and writes `run.json`, `state.json` and `loss.svg`. This can take several seconds during a save. A forced kill/power loss cannot execute this cleanup; resume the last successfully saved update instead.

`--max-seconds` is a soft **per-invocation training-loop budget**, including evaluation and checkpoint saves, checked between updates. Setup and final export/generation are outside that budget, and an in-flight save can overrun it. `--stop-after N` is an absolute update boundary for a planned earlier stop. Neither inserts an extra validation measurement or alters incomplete averaging windows.

The hardware-monitor wrapper also allows a cooperative child up to 120 seconds to exit after a stop signal (`--stop-timeout` overrides this), before forced termination. See [hardware monitoring](MONITORING.md) for its command wrapper and raw sensor logs.

## Resume with one continuous graph

Keep the same data, architecture, seed, batch size, optimizer/evaluation settings and `--log-every`. Choose a fresh child output directory and resume from the saved latest checkpoint:

```bash
uv run --locked --group experiments --group cosmopedia python experiments/cosmopedia/train.py \
  --device cuda --session experiments/cosmopedia/artifacts/my-session \
  --output experiments/cosmopedia/artifacts/my-session/part-02 \
  --resume experiments/cosmopedia/artifacts/my-session/part-01/latest.pt \
  --steps 100000 --eval-every 1000 --checkpoint-every 5000 --log-every 100 \
  --max-seconds 7200
```

The same browser URL switches to the new output directory and displays restored history plus new measurements. The checkpoint is the authority: measurements beyond the selected checkpoint are not spliced into the resumed trajectory. Resuming an older best checkpoint intentionally forks from that earlier update. This prevents duplicate update numbers and misleading future points from abandoned work. Historical weights-only files still cannot resume.

One trainer at a time may own a session; a file lock rejects concurrent writers. Atomic JSON replacement prevents partial browser reads. A failed Python process marks its last published state as failed; a hard-killed process is detected by the server when its PID disappears. The local server serves only the dashboard/state/graph and a narrowly scoped stop request; it does not serve model checkpoints or arbitrary files. Stop requests require a same-host JSON request.

The graph survives even with the server stopped: open the run's `loss.svg`, or restart the dashboard against the same session folder. The running view reads `state.json`; frozen graphs and reports remain separate for each invocation. Training and dashboard processes are independent, so closing a browser tab does not stop training.

## The November 14 longer experiment

The [recorded plan](results/long-10m-setup/plan.json) uses **10M training tokens**, seed 7, batch 4, context 256 and the same 15.8M-parameter architecture. The [verification record](results/long-10m-setup/data-verification.json) checks every document hash and reconstructed input/target tensor. Prepared files total **249.34 MB**, with 13,960 training documents and 46,058 windows. Validation is the **same 89 documents / 300 windows / 65,536 tokens** as the original sample, verified tensor-for-tensor. This run evaluates all 300 windows rather than the original 32-window diagnostic, so its losses are not an identical evaluation procedure.

The larger data identity starts a new randomly initialized experiment; the small baseline and its reports remain intact. It does not silently warm-start/resume the old 1M-token experiment with different data.

Settings:

| Setting | Value |
| --- | ---: |
| Total target updates | 500,000 |
| Combined calibration + continuation loop budget | Approximately 7,200 seconds |
| Training-loss point | Every 100 updates |
| Full fixed validation | Every 1,000 updates |
| Periodic latest checkpoint | Every 5,000 updates, plus graceful stop |
| Historical best checkpoint | On validation improvement |

The first **1,000 updates completed and stopped safely**. Validation fell from **10.8514 to 6.2993**. The measured loop took **13.16 seconds**, approximately **66,477 non-padding targets/sec** with this save cadence. The [unchanged calibration report](results/long-10m-setup/calibration-run.json), [saved graph](results/long-10m-setup/calibration-loss.svg) and [dashboard screenshot](results/long-10m-setup/dashboard.png) preserve that check. The continuation resumed update 1,000, including its optimizer/RNG/curve state, with the remaining time budget. The continuation has now completed 500,000 updates; see the [completed results and charts](results/long-10m-run/README.md).

At the calibration's sampled-target rate, 500,000 updates would process roughly 437M targets with replacement. That repeats a 10M-token sample; it is not 437M distinct tokens or a guaranteed number of complete epochs. Watch training and validation together for overfitting. The historical best snapshot preserves the best measured validation result even if later training gets worse. The time/update caps do not promise a useful model or fluent generation.

On this host, the dashboard and continuation can run as transient user services so they survive the chat/terminal ending. They are not enabled at boot. The live session folder is `experiments/cosmopedia/artifacts/long-10m-20261114`; run directories and sensor observations remain ignored. Check service state/logs with:

```bash
systemctl --user status learning-llms-loss-dashboard.service learning-llms-long-training.service
journalctl --user -u learning-llms-long-training.service --no-pager
```

Use the dashboard's stop button to save and stop training while keeping the graph available. To shut down the dashboard itself:

```bash
systemctl --user stop learning-llms-loss-dashboard.service
```

Tests exercise the real HTTP stop endpoint and fresh training processes, exact uninterrupted-versus-resumed state/curve equality, signal/time-budget stops, a stop between averaging boundaries, single-writer locking and path/origin restrictions. No AWS resources are involved.
