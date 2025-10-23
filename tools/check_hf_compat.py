# Small compatibility check for huggingface_hub and sentence-transformers
import importlib

def check():
    out = {}
    try:
        hf = importlib.import_module('huggingface_hub')
        out['huggingface_hub_version'] = getattr(hf, '__version__', 'unknown')
        out['has_cached_download'] = hasattr(hf, 'cached_download')
    except Exception as e:
        out['huggingface_hub_error'] = str(e)
    try:
        st = importlib.import_module('sentence_transformers')
        out['sentence_transformers_version'] = getattr(st, '__version__', 'unknown')
    except Exception as e:
        out['sentence_transformers_error'] = str(e)
    return out

if __name__ == '__main__':
    import json
    print(json.dumps(check(), indent=2))
