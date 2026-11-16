"""A local live loss dashboard and persistent, atomic training progress files.

No network dependencies: the browser draws SVG from a small JSON endpoint.
Session folders point to one active run; resumed runs import checkpoint history.
"""

import argparse
import atexit
import fcntl
import html
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def atomic_json(path: Path, value: dict) -> None:
    """Replace a JSON document without exposing partial writes to the browser."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def loss_svg(state: dict) -> str:
    """Build a portable loss graph with global update numbers, using no plotting RNG."""
    train = [(row["step"], row["loss"]) for row in state.get("training_curve", [])]
    validation = [
        (row["step"], row["validation_loss"]) for row in state.get("history", [])
    ]
    pending = state.get("pending_training_point")
    points = (
        train + validation + ([(pending["step"], pending["loss"])] if pending else [])
    )
    finite = [(x, y) for x, y in points if math.isfinite(y)]
    xmax = max([1, state.get("step", 0), *[p[0] for p in finite]])
    low = min([y for _, y in finite], default=0)
    high = max([y for _, y in finite], default=1)
    pad = max(0.1, (high - low) * 0.08)
    low, high = low - pad, high + pad

    def coordinates(values: list[tuple[int, float]]) -> str:
        return " ".join(
            f"{70 + x / xmax * 1080:.2f},{440 - (y - low) / (high - low) * 380:.2f}"
            for x, y in values
            if math.isfinite(y)
        )

    marks: list[str] = []
    for index in range(6):
        y = 440 - index * 76
        x = 70 + index * 216
        marks.append(
            f'<path d="M70 {y}H1150 M{x} 60V440" stroke="#e2e8f0"/><text x="60" y="{y + 4}" text-anchor="end">{low + (high - low) * index / 5:.2f}</text><text x="{x}" y="465" text-anchor="middle">{xmax * index / 5:.0f}</text>'
        )
    if pending:
        marks.append(
            f'<circle cx="{coordinates([(pending["step"], pending["loss"])]).split(",")[0]}" cy="{coordinates([(pending["step"], pending["loss"])]).split(",")[1]}" r="4" fill="white" stroke="#2563eb"/>'
        )
    title = f"Loss through update {state.get('step', 0):,} · {state.get('status', 'waiting')}"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 510" role="img" aria-label="Training and validation loss"><rect width="1200" height="510" fill="white"/><g font-family="system-ui" font-size="13" fill="#334155"><text x="70" y="28" font-size="20">{html.escape(title)}</text><text x="70" y="50" fill="#2563eb">Recent training window</text><text x="310" y="50" fill="#d97706">Fixed-window validation</text>{"".join(marks)}<polyline points="{coordinates(train)}" fill="none" stroke="#2563eb" stroke-width="2"/><polyline points="{coordinates(validation)}" fill="none" stroke="#d97706" stroke-width="2"/>{"".join(f'<circle cx="{70 + x / xmax * 1080:.2f}" cy="{440 - (y - low) / (high - low) * 380:.2f}" r="3" fill="#d97706"/>' for x, y in validation)}<text x="610" y="500" text-anchor="middle">Global optimizer update</text></g></svg>'''


class LiveLog:
    """Keep the UI current and preserve a graph on a normal stop or failure."""

    def __init__(self, output: Path, session: Path | None) -> None:
        self.output = output
        self.state: dict = {}
        self.lock = None
        if session is not None:
            session = session.resolve()
            relative = output.resolve().relative_to(session)
            session.mkdir(parents=True, exist_ok=True)
            self.lock = (session / ".training.lock").open("a")
            try:
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self.lock.close()
                raise ValueError(
                    "Another trainer already owns this live session"
                ) from error
            atomic_json(session / "active.json", {"run": str(relative)})
        atexit.register(self.on_exit)

    def update(self, **values: object) -> None:
        """Publish a checkpoint-derived cumulative curve and current invocation status."""
        self.state.update(values)
        self.state.update(
            pid=os.getpid(),
            updated_at_unix=time.time(),
            run_directory=str(self.output.resolve()),
        )
        atomic_json(self.output / "state.json", self.state)

    def finish(self, status: str, **values: object) -> None:
        """Freeze the observed progress and an SVG for viewing without the server."""
        self.update(status=status, **values)
        (self.output / "loss.svg").write_text(loss_svg(self.state))

    def on_exit(self) -> None:
        """Mark ordinary Python failures; SIGKILL cannot execute cleanup."""
        if self.state and self.state.get("status") not in [
            "stopped",
            "completed",
            "failed",
        ]:
            self.finish(
                "failed", stop_reason="process exited before normal finalization"
            )


def active_run(session: Path) -> Path | None:
    """Resolve only a descendant run directory from the session pointer."""
    pointer = session / "active.json"
    if not pointer.exists():
        return None
    run = (session / json.loads(pointer.read_text())["run"]).resolve()
    if run == session.resolve() or not run.is_relative_to(session.resolve()):
        raise ValueError("Invalid active run path")
    return run


PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Live training loss</title>
<style>body{font:16px system-ui;color:#1e293b;background:#f8fafc;max-width:1300px;margin:32px auto;padding:0 20px}button,a{margin-right:12px}button{padding:10px 16px;cursor:pointer}#plot{width:100%;background:white;border:1px solid #e2e8f0}pre{white-space:pre-wrap}#status{font-size:20px}small{color:#475569}table{border-collapse:collapse}td,th{padding:6px 18px;border-bottom:1px solid #ddd;text-align:left}</style>
<h1>Live training loss</h1><p id="status">Waiting for a trainer…</p><button id="stop">Stop &amp; save checkpoint</button><a href="/loss.svg" download="loss.svg">Download current graph</a><p><small>Refreshes every 2 seconds. Curves use global update numbers and restored checkpoint history. A stop waits for a safe update/checkpoint boundary.</small></p><div id="plot"></div><p id="detail"></p><details><summary>Recent validation measurements</summary><table id="history"></table></details>
<script>
const status=document.querySelector('#status'),stop=document.querySelector('#stop');
async function refresh(){try{const response=await fetch('/state',{cache:'no-store'});const s=await response.json();status.textContent=`${s.status} · update ${s.step??0} / ${s.planned_steps??'—'} · best validation ${s.best_validation_loss?.toFixed(4)??'—'} at ${s.best_step??'—'}`;stop.disabled=!['running','saving','initializing','stopping'].includes(s.status);document.querySelector('#detail').textContent=`${s.run_directory??''} | Last saved update: ${s.latest_checkpoint_step??'none'} | Resumed at: ${s.segment_start_step??0} | ${s.stop_reason??''}`;const plot=await fetch('/loss.svg',{cache:'no-store'});document.querySelector('#plot').innerHTML=await plot.text();document.querySelector('#history').innerHTML='<tr><th>Update</th><th>Recent training</th><th>Validation</th></tr>'+(s.history??[]).slice(-15).reverse().map(r=>`<tr><td>${r.step}</td><td>${r.train_batch_loss?.toFixed(4)??'—'}</td><td>${r.validation_loss.toFixed(4)}</td></tr>`).join('');}catch(e){status.textContent='Dashboard disconnected; training may still be running.';}}
stop.onclick=async()=>{stop.disabled=true;const r=await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});if(!r.ok)alert(await r.text());await refresh();};refresh();setInterval(refresh,2000);
</script></html>"""


def serve(session: Path, port: int) -> None:
    """Serve this one session on loopback, with a narrowly scoped stop endpoint."""
    session.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def allowed(self) -> bool:
            expected = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in expected and (
                origin is None or origin in {f"http://{host}" for host in expected}
            )

        def send(self, body: str, content_type: str, status: int = 200) -> None:
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def state(self) -> dict:
            run = active_run(session)
            if run is None or not (run / "state.json").exists():
                return {"status": "waiting"}
            result = json.loads((run / "state.json").read_text())
            if result.get("status") in [
                "running",
                "saving",
                "stopping",
                "initializing",
            ]:
                try:
                    os.kill(result["pid"], 0)
                except ProcessLookupError:
                    result.update(
                        status="process exited",
                        stop_reason="Last recorded state; resume from a saved checkpoint",
                    )
            return result

        def do_GET(self) -> None:
            if not self.allowed():
                self.send("Local dashboard host/origin required", "text/plain", 403)
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send(PAGE, "text/html; charset=utf-8")
            elif path == "/state":
                self.send(json.dumps(self.state()), "application/json")
            elif path == "/loss.svg":
                self.send(loss_svg(self.state()), "image/svg+xml")
            else:
                self.send("Not found", "text/plain", 404)

        def do_POST(self) -> None:
            if (
                not self.allowed()
                or self.headers.get("Content-Type") != "application/json"
            ):
                self.send("Local JSON request required", "text/plain", 403)
                return
            if urlsplit(self.path).path != "/stop":
                self.send("Not found", "text/plain", 404)
                return
            run = active_run(session)
            if run is None or self.state().get("status") not in [
                "running",
                "saving",
                "stopping",
                "initializing",
            ]:
                self.send("No running trainer", "text/plain", 409)
                return
            (run / "STOP").touch()
            self.send('{"stop_requested":true}', "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    atomic_json(
        session / "dashboard.json",
        {"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()},
    )
    print(f"Live loss dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    serve(options.session.resolve(), options.port)
