import os
import argparse

import pytest

def get_tests():
    test_files = [file for file in os.listdir("pytest_test") if len(file) > 4 and file[:4] == "test"]
    return test_files

def run_tests(all, until):
    test_files = get_tests()
    n = len(test_files)

    if until is None:
        until = n
    else:
        n = min(until, n)

    test_in_order = [f"pytest_test/test_task{i}.py" for i in range(1, n+1)]

    if all:
        pytest.main(test_in_order + ["--tb=short"])
    else:
        pytest.main(test_in_order + ["--tb=short", "-x"])

def run_test(n):
    test_files = get_tests()
    test = f"test_task{n}.py"

    if test not in test_files:
        message = f"Test {n} does not exist"
        raise ValueError(message)

    pytest.main(["pytest_test/"+test] + ["--tb=short", "-x"])


def main():
    parser = argparse.ArgumentParser(description="Run tests in order")

    parser.add_argument("--only", "-o", type=int, help="Run a specific test")
    parser.add_argument("--until", "-u", type=int, default=None, help="Run tests until a specific test")
    parser.add_argument("--all", "-a", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.only:
        run_test(args.only)
    else:
        run_tests(args.all, args.until)



if __name__ == "__main__":
    main()