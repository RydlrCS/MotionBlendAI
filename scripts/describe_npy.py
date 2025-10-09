#!/usr/bin/env python3
"""
Read a .npy file and print JSON with shape, dtype and a small sample (first frame first few values).
Usage: describe_npy.py /absolute/path/to/file.npy
"""
import sys, json
import numpy as np
from typing import Any

def describe(path: str) -> dict[str, Any]:
    try:
        arr = np.load(path)
    except Exception as e:
        return {"error": str(e)}
    info: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    try:
        # sample first frame or flattened first 10 numbers
        if arr.size == 0:
            info["sample"] = []
        else:
            flat = arr.flatten()
            info["sample"] = flat[:10].tolist()
    except Exception as e:
        info["sample_error"] = str(e)
    return info

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({"error":"missing path"}))
        sys.exit(2)
    out = describe(sys.argv[1])
    print(json.dumps(out))
