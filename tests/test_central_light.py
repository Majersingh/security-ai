"""Central must import with the module's heavy stack UNAVAILABLE.

Why this exists: on a dev box both requirement sets share one virtualenv, so a
stray `import cv2` or `import torch` in central code works fine locally and only
fails when central is deployed to a small GPU-less VM — a silent mistake that
surfaces at the worst possible moment.

This blocks the heavy packages at import time and then imports central, which is
the closest we can get to "pretend we're on the deployment box".
"""
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Everything central must never need. These belong to the analysis module.
FORBIDDEN = ("torch", "ultralytics", "cv2", "supervision", "av", "pandas", "numpy")


class _Blocker:
    """Raise ImportError for the heavy stack, as if it weren't installed."""

    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in FORBIDDEN:
            raise ImportError(
                f"'{name}' is not available to the central app — central must "
                f"deploy without the CV stack (no GPU, no torch). Move whatever "
                f"needs it into module/."
            )
        return None


def main() -> int:
    # Drop anything already imported so the blocker actually gets consulted.
    for mod in list(sys.modules):
        if mod.split(".")[0] in FORBIDDEN:
            del sys.modules[mod]
    for mod in list(sys.modules):
        if mod.startswith("central"):
            del sys.modules[mod]

    sys.meta_path.insert(0, _Blocker())
    ok = True
    try:
        import central.app          # noqa: F401
        import central.db           # noqa: F401
        import central.module_client  # noqa: F401
        import central.placement    # noqa: F401
        print("  central imports with torch/cv2/numpy/... blocked ... PASS")
    except ImportError as exc:
        print(f"  central imports with heavy stack blocked ....... FAIL\n     {exc}")
        ok = False
    finally:
        sys.meta_path.pop(0)

    # Sanity: the blocker really would have caught something.
    sys.meta_path.insert(0, _Blocker())
    try:
        import numpy  # noqa: F401
        print("  blocker itself works ........................... FAIL (numpy imported)")
        ok = False
    except ImportError:
        print("  blocker itself works ........................... PASS")
    finally:
        sys.meta_path.pop(0)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
