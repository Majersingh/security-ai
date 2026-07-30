"""The streamer must import with the CV stack UNAVAILABLE.

A video host installs `streamer/requirements.txt` — no torch, no ultralytics, no
supervision. On a dev box everything shares one virtualenv, so a stray import would
work locally and only fail on the video-only box it was built for.

Mirrors tests/test_central_light.py, which does the same for central.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "streamer"))

# The streamer decodes and encodes; it must never need the model stack.
FORBIDDEN = ("torch", "ultralytics", "supervision", "pandas")


class _Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in FORBIDDEN:
            raise ImportError(
                f"'{name}' is not available to the streamer — it deploys without the "
                f"CV stack (no GPU, no torch). Anything needing it belongs in module/."
            )
        return None


def main() -> int:
    for mod in list(sys.modules):
        root = mod.split(".")[0]
        if root in FORBIDDEN or root in ("videostream", "config", "procutil", "sources"):
            del sys.modules[mod]

    sys.meta_path.insert(0, _Blocker())
    ok = True
    try:
        import config          # noqa: F401  streamer/config.py
        import procutil        # noqa: F401  core/procutil.py
        import sources         # noqa: F401  core/sources.py
        import videostream     # noqa: F401  streamer/videostream.py
        print("  streamer core imports with torch/ultralytics blocked ... PASS")
    except ImportError as exc:
        print(f"  streamer core imports with CV stack blocked ......... FAIL\n     {exc}")
        ok = False
    finally:
        sys.meta_path.pop(0)

    # The blocker must actually be capable of failing, or the check above proves nothing.
    sys.meta_path.insert(0, _Blocker())
    try:
        import torch  # noqa: F401
        print("  blocker itself works ............................... FAIL")
        ok = False
    except ImportError:
        print("  blocker itself works ............................... PASS")
    finally:
        sys.meta_path.pop(0)

    # And the streamer's own config must not be the module's heavyweight Config.
    from config import StreamerConfig
    cfg = StreamerConfig.from_env()
    lean = not hasattr(cfg, "model_path") and not hasattr(cfg, "inference_imgsz")
    print(f"  own config, not the module's ....................... "
          f"{'PASS' if lean else 'FAIL'}  (max_streams={cfg.streamer_max_streams})")
    ok &= lean

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
