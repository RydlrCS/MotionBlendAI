import os
import sys
import json

import tempfile

def setup_module():
    # ensure repo root on path so imports work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


from importlib import import_module

search_mod = import_module('project.search_api.search_service')
app = search_mod.app


class DummyES:
    def __init__(self):
        self._info = {"version": "dummy"}
        self.calls = []

    def ping(self):
        return True

    def info(self):
        return self._info

    class indices:
        @staticmethod
        def exists(index):
            return True

        @staticmethod
        def create(index, body=None):
            return None

        @staticmethod
        def refresh(index):
            return None

    def index(self, index, id, body):
        self.calls.append((index, id, body))

    def search(self, index, body):
        # return a deterministic response for tests
        return {"hits": {"hits": [{"_id": "1", "_source": {"description": "dummy"}}]}}


def test_health_endpoint():
    client = app.test_client()
    # inject dummy ES
    app.config['ES_CLIENT'] = DummyES()
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'ok'
    assert data['elasticsearch'] is not None


def test_seed_endpoint_monkeypatched(monkeypatch):
    client = app.test_client()
    called = {}

    def fake_seed(folder, es_client=None, max_workers=1, max_files=None, dry_run=False):
        called['folder'] = folder
        called['es_client'] = es_client

    # monkeypatch the seeder module by ensuring the `scripts` package points to a module
    import types
    fake_mod = types.SimpleNamespace(seed_from_folder=fake_seed)
    # Ensure there is a `scripts` package module in sys.modules and attach the fake submodule
    scripts_mod = sys.modules.get('scripts')
    if scripts_mod is None:
        scripts_mod = types.ModuleType('scripts')
        monkeypatch.setitem(sys.modules, 'scripts', scripts_mod)
    # set the submodule both as package attribute and in sys.modules
    setattr(scripts_mod, 'seed_motions', fake_mod)
    monkeypatch.setitem(sys.modules, 'scripts.seed_motions', fake_mod)

    # inject dummy ES so get_es_client works
    app.config['ES_CLIENT'] = DummyES()
    resp = client.post('/seed', json={'folder': 'project/seed_motions'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'seeded'
    assert called.get('folder') == 'project/seed_motions'


def test_search_endpoint_uses_es():
    client = app.test_client()
    app.config['ES_CLIENT'] = DummyES()
    resp = client.post('/search', json={'text_query': 'walk', 'k': 1})
    assert resp.status_code == 200
    data = resp.get_json()
    # should return a list of hits
    assert isinstance(data, list)
    assert len(data) == 1
