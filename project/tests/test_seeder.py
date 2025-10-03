import os
from importlib import import_module
import pytest


class DummyIndices:
    def __init__(self):
        self._exists: dict[str, bool] = {}

    def exists(self, index):
        return self._exists.get(index, False)

    def create(self, index, body=None):
        self._exists[index] = True

    def refresh(self, index):
        # no-op for dummy
        return True


class DummyES:
    def __init__(self):
        self.indices = DummyIndices()
        self._docs = {}

    def ping(self):
        return True

    def index(self, index, id, body):
        self._docs[id] = body


def test_seeder_indexes_files_using_repo_samples():
    # Use the repository's sample folder `project/seed_motions` for testing.
    repo_root = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.normpath(os.path.join(repo_root, 'seed_motions'))

    if not os.path.isdir(folder):
        pytest.skip(f'sample folder not found: {folder}')

    # discover files with supported extensions
    supported = {'.fbx', '.bvh', '.trc'}
    motion_files = [
        os.path.join(r, f)
        for r, _, files in os.walk(folder)
        for f in files
        if os.path.splitext(f)[1].lower() in supported
    ]

    if not motion_files:
        pytest.skip(f'no sample motion files found under {folder}')

    seed_mod = import_module('scripts.seed_motions')
    dummy = DummyES()

    # Limit to a small number for deterministic test runtime
    max_files = 3
    seed_mod.seed_from_folder(folder=folder, es_client=dummy, dry_run=False, max_workers=2, max_files=max_files)

    # Expect number of indexed docs == min(max_files, available files)
    expected_count = min(max_files, len(motion_files))
    assert len(dummy._docs) == expected_count
