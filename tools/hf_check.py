import importlib, json
res={}
for mod in ('huggingface_hub','sentence_transformers'):
    try:
        m=importlib.import_module(mod)
        res[mod]={'version':getattr(m,'__version__',None)}
    except Exception as e:
        res[mod]={'error':str(e)}
if 'huggingface_hub' in res and 'error' not in res['huggingface_hub']:
    import huggingface_hub as hf
    res['huggingface_hub']['has_cached_download']=hasattr(hf,'cached_download')
print(json.dumps(res))
