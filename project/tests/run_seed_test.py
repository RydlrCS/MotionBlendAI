import os
import tempfile
import sys

from importlib import import_module


def run_test():
    # ensure repo root is on sys.path so we can import top-level scripts
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    seed_mod = import_module('scripts.seed_motions')

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

    tmpdir = tempfile.mkdtemp()
    folder = os.path.join(tmpdir, 'seed_motions')
    os.makedirs(folder, exist_ok=True)
    # create files
    with open(os.path.join(folder, 'walk_1.bvh'), 'w') as f:
        f.write('HIERARCHY\nROOT Hips\n')
    with open(os.path.join(folder, 'run_1.trc'), 'w') as f:
        f.write('Frames\n1 0.1 0.2 0.3')
    with open(os.path.join(folder, 'jump_1.fbx'), 'wb') as f:
        f.write(b'FBX binary...')

    dummy = DummyES()

    # run seeder using DI (es_client)
    seed_mod.seed_from_folder(folder, es_client=dummy, max_workers=2)

    print('Indexed calls:', len(dummy.calls))
    for c in dummy.calls:
        print(c[1])

    assert len(dummy.calls) == 3


if __name__ == '__main__':
    try:
        run_test()
        print('SEED TEST PASSED')
    except AssertionError:
        print('SEED TEST FAILED', file=sys.stderr)
        sys.exit(2)