#!/usr/bin/env python3
import requests
import time
import sys


def wait_for_health(url='http://localhost:8080/health', timeout=60):
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                j = r.json()
                if j.get('status') == 'ok':
                    print('search service healthy')
                    return True
        except Exception:
            pass
        time.sleep(1)
    print('timeout waiting for search service')
    return False


def seed_via_api(folder='project/seed_motions', max_files=5):
    url = 'http://localhost:8080/seed'
    r = requests.post(url, json={'folder': folder, 'max_files': max_files})
    print('seed resp', r.status_code, r.text[:200])
    return r.status_code == 200


def check_es_count(min_count=1):
    url = 'http://localhost:9200/motions/_count'
    for _ in range(20):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                c = r.json().get('count', 0)
                print('es count', c)
                if c >= min_count:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def check_vector_shape():
    url = 'http://localhost:9200/motions/_search?size=1'
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print('search failed', r.status_code, r.text[:200])
            return False
        hits = r.json().get('hits', {}).get('hits', [])
        if not hits:
            print('no hits')
            return False
        src = hits[0].get('_source', {})
        vec = src.get('motion_vector') or src.get('motionVector')
        if not isinstance(vec, list):
            print('vector missing or not a list')
            return False
        print('vector length', len(vec))
        return len(vec) == 384
    except Exception as e:
        print('exception checking vector', e)
        return False


def main():
    if not wait_for_health():
        sys.exit(2)
    # seed a limited number of files
    if not seed_via_api(max_files=5):
        sys.exit(3)
    # wait until docs appear
    if not check_es_count(min_count=1):
        print('documents not indexed')
        sys.exit(4)
    if not check_vector_shape():
        print('vector shape check failed')
        sys.exit(5)
    print('integration checks passed')
    sys.exit(0)


if __name__ == '__main__':
    main()
