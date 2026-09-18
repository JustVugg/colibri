import json
import os


def discover_gpus():
    return json.loads(os.environ.get("FAKE_GPU_DATA", "[]"))
