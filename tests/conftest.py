import os

import pytest


def run_tests():
    test_files = ['/test_layers.py', '/test_rand.py']
    for test_file in test_files:
        pytest.main([os.getcwd() + test_file])


if __name__ == '__main__':
    run_tests()
