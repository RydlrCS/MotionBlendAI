"""
Simple verification helper for orchestration configuration.

Checks:
- Required environment variables
- pipeline.yaml existence on local path or GCS (if PIPELINE_YAML starts with gs://)
- presence of Cloud Function entry points in trigger file

This script is safe to run locally and will not modify resources.
"""
import os
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


REQUIRED_ENVS = [
    "PROJECT",
    "LOCATION",
    "PIPELINE_NAME",
    "PIPELINE_ROOT",
    "PIPELINE_YAML",
]


def check_envs():
    missing = [k for k in REQUIRED_ENVS if not os.getenv(k)]
    if missing:
        print("MISSING_ENV_VARS:", missing)
        return False
    print("All required environment variables present")
    return True


def check_pipeline_yaml(path: str):
    if path.startswith("gs://"):
        # check that gsutil is available and that the object exists (best-effort)
        import shutil
        gsutil = shutil.which("gsutil")
        if not gsutil:
            print("PIPELINE_YAML is a GCS path but 'gsutil' is not available on PATH. Skipping remote check.")
            return True
        # run a lightweight check
        from subprocess import run, PIPE
        res = run([gsutil, "ls", path], stdout=PIPE, stderr=PIPE, text=True)
        if res.returncode != 0:
            print("GCS_LOOKUP_FAILED:", res.stderr.strip())
            return False
        print("pipeline.yaml found in GCS:", path)
        return True
    else:
        p = Path(path)
        if p.exists():
            print("Local pipeline.yaml found:", str(p))
            return True
        print("Local pipeline.yaml not found:", str(p))
        return False


def check_trigger_module(trigger_path: str):
    p = Path(trigger_path)
    if not p.exists():
        print("Trigger file not found:", trigger_path)
        return False
    try:
        spec = spec_from_file_location("trigger_mod", str(p))
        if spec is None or spec.loader is None:
            print("Failed to create module spec or loader is missing for:", trigger_path)
            return False
        mod = module_from_spec(spec)
        # spec.loader is not None here because of the check above
        spec.loader.exec_module(mod)
    except Exception as e:
        print("Failed to import trigger module:", e)
        return False
    # check expected callable names
    expected = ["trigger_vertex_pipeline", "http_trigger"]
    missing = [name for name in expected if not hasattr(mod, name)]
    if missing:
        print("Trigger module missing expected symbols:", missing)
        return False
    print("Trigger module contains expected symbols")
    return True


def main():
    ok = True
    print("Checking required environment variables...")
    if not check_envs():
        ok = False

    pipeline_yaml = os.getenv("PIPELINE_YAML", "project/vertex_pipeline/pipeline.yaml")
    print("Checking pipeline yaml path ->", pipeline_yaml)
    if not check_pipeline_yaml(pipeline_yaml):
        ok = False

    trigger_file = os.getenv("TRIGGER_FILE", "project/vertex_pipeline/trigger_pipeline_function.py")
    print("Checking trigger file ->", trigger_file)
    if not check_trigger_module(trigger_file):
        ok = False

    if not ok:
        print("VERIFICATION_FAILED: see messages above")
        sys.exit(2)
    print("VERIFICATION_OK: orchestration basic checks passed")


if __name__ == "__main__":
    main()
