"""Optional JSONL phase markers sharing the monitor's monotonic system clock."""

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Events:
    """Append flushed markers when COSMOPEDIA_EVENTS is set; otherwise do nothing."""

    def __init__(self) -> None:
        value = os.environ.get("COSMOPEDIA_EVENTS")
        self.path = Path(value) if value else None

    def emit(self, phase: str, kind: str, **values: object) -> None:
        """Timestamp an event; monitoring must not consume any training RNG."""
        if self.path is not None:
            with self.path.open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "monotonic_ns": time.monotonic_ns(),
                            "phase": phase,
                            "kind": kind,
                            **values,
                        }
                    )
                    + "\n"
                )

    @contextmanager
    def phase(self, name: str, **values: object) -> Iterator[None]:
        """Bracket synchronous work, retaining an end marker even on failure."""
        self.emit(name, "start", **values)
        try:
            yield
        except BaseException:
            self.emit(name, "end", status="failed", **values)
            raise
        else:
            self.emit(name, "end", status="ok", **values)


def estimated_training_flops(config: dict[str, int], batch: int, length: int) -> int:
    """Approximate 3x forward matmul FLOPs, counting a multiply-add as two.

    Includes Q/K/V/output projections, two feedforward matrices, vocabulary
    projection and full-square attention matmuls. Real causal/fused kernels may
    do less work. Excludes embeddings, norms, GELU, softmax, loss and optimizer.
    This is an algorithmic estimate, not a hardware performance-counter reading.
    """
    d, h, layers, vocab = (
        config[k] for k in ["width", "hidden_width", "layers", "vocabulary_size"]
    )
    forward = 2 * batch * length * (layers * (4 * d * d + 2 * d * h) + d * vocab)
    forward += 4 * batch * layers * length * length * d
    return 3 * forward
