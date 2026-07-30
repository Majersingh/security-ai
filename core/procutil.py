"""Process-level helpers shared by every deployable.

Lives in the top-level ``core/`` package because the detection module and the
streamer both need them, and neither should import the other's tree. Deliberately
tiny and dependency-free (stdlib + an optional cv2/torch touch) so the streamer can
install without the CV stack.
"""

from __future__ import annotations

import logging
import os
import sys

# Every library that spawns its own intra-op thread pool from an env var. These
# must be set *before* the library is imported to take effect, which is why
# `limit_process_threads` is called at the very top of each process entry point.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package logger.

    Idempotent: repeated calls will not add duplicate handlers.
    """
    logger = logging.getLogger("operator_monitor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def physical_cores() -> int:
    """Number of *physical* cores (hyperthread siblings collapsed).

    Sizing process pools by ``os.cpu_count()`` oversubscribes on any SMT host:
    two hyperthreads on one core do not give two cores' worth of the vectorised
    decode/annotate work this pipeline does. Falls back to the logical count on
    platforms without ``/proc/cpuinfo``.
    """
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():              # blank line ends a cpu block
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return os.cpu_count() or 1


def limit_process_threads(cv_threads: int = 1, torch_threads: int = 1) -> None:
    """Pin this process's intra-op thread pools so N processes don't fight.

    Every worker process runs the same OpenCV + torch stack, and each one
    defaults to sizing its own pool from the *machine's* core count — so N
    processes each claim the whole box and the scheduler thrashes. Since
    parallelism here comes from running many processes (one per group of feeds),
    each process wants ~1 compute thread, not ``ncpu`` of them.

    Call this **first**, before importing torch/cv2, so the env vars are seen at
    import time; the explicit setters below cover the case where something
    already pulled them in.
    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(max(1, torch_threads)))
    try:
        import cv2

        cv2.setNumThreads(max(1, cv_threads))
    except Exception:  # noqa: BLE001 - thread capping is best-effort, never fatal
        pass
    if "torch" in sys.modules:          # only if already imported; env covers the rest
        try:
            sys.modules["torch"].set_num_threads(max(1, torch_threads))
        except Exception:  # noqa: BLE001
            pass
