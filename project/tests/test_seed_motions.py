import os
import tempfile
from unittest import mock

import pytest

# import the module under test
import importlib

seed_mod = importlib.import_module('scripts.seed_motions')


class DummyIndices:
    def __init__(self):
        self._created = False

    def exists(self, index):
        return self._created

    def create(self, index, body=None):
        self._created = True


class DummyES:
    def __init__(self):
        self.calls = []
        self.indices = DummyIndices()

    def ping(self):
        return True

    def index(self, index, id, body):
        self.calls.append((index, id, body))

    def indices_refresh(self, index):
        pass


def test_seed_from_folder_with_mock_es(tmp_path, monkeypatch):
    # create a temporary sample folder with a few fake files
    folder = tmp_path / 'samples'
    folder.mkdir()
    # create a BVH (text), a TRC (text), and a binary-like FBX file
    bvh = folder / 'walk_1.bvh'
    bvh.write_text('HIERARCHY\nROOT Hips\n...')
    trc = folder / 'run_1.trc'
    trc.write_text('Frames\n1 0.1 0.2 0.3')
    fbx = folder / 'jump_1.fbx'
    fbx.write_bytes(b'Kaydara FBX binary...')

    dummy = DummyES()

    # monkeypatch the module's es with our dummy
    monkeypatch.setattr(seed_mod, 'es', dummy)

    # run seeder (should not raise)
    seed_mod.seed_from_folder(str(folder))

    # assert that three documents were indexed
    # (doc ids are relative paths)
    assert len(dummy.calls) == 3
    ids = {c[1] for c in dummy.calls}
    assert 'walk_1.bvh' in ids
    assert 'run_1.trc' in ids
    assert 'jump_1.fbx' in ids
