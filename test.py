import os

import pytest

def run_tests():
    test_files = [file for file in os.listdir("pytest_test") if len(file) > 4 and file[:4] == "test"]
    n = len(test_files)

    test_in_order = [f"pytest_test/test_task{i}.py" for i in range(1, n+1)]

    pytest.main(test_in_order + ["--tb=short", "-x"])

if __name__ == "__main__":
    run_tests()