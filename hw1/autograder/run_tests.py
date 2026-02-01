#!/usr/bin/env python3
"""
Unified Gradescope-style autograder for BOTH FCN and CNN.

Design philosophy (mirrors MP-style autograder):
- Run every test independently
- Never crash entire script
- Per-test timeout
- Partial credit
- Gradescope-compatible JSON
- Clear failure messages
"""

import json
import traceback
import argparse
import func_timeout

from tests_fcn import ALL_TESTS as FCN_TESTS
from tests_cnn import ALL_TESTS as CNN_TESTS


# ======================================================
# Helpers
# ======================================================

def fail(message):
    return {
        "score": 0,
        "output": message,
        "visibility": "visible"
    }


def make_result(name, score, max_score, output=""):
    return {
        "name": name,
        "score": score,
        "max_score": max_score,
        "output": output,
        "visibility": "visible"
    }


# ======================================================
# Runner
# ======================================================

def run_test(suite_name, test_name, points, fn, timeout):
    """
    Runs a single test safely with timeout and exception handling.
    Returns result dict.
    """

    full_name = f"{suite_name} :: {test_name}"

    try:
        func_timeout.func_timeout(timeout, fn)

        return make_result(
            name=full_name,
            score=points,
            max_score=points,
            output="Passed"
        )

    except func_timeout.FunctionTimedOut:
        return make_result(
            name=full_name,
            score=0,
            max_score=points,
            output="Timed out (likely infinite loop or very slow implementation)"
        )

    except Exception:
        return make_result(
            name=full_name,
            score=0,
            max_score=points,
            output=traceback.format_exc()
        )


# ======================================================
# Main
# ======================================================

def main(args):

    suites = [
        ("FCN", FCN_TESTS),
        ("CNN", CNN_TESTS),
    ]

    results = []
    total_score = 0

    for suite_name, suite in suites:
        for name, points, fn in suite:
            result = run_test(suite_name, name, points, fn, args.timeout)
            results.append(result)
            total_score += result["score"]

            if not args.gradescope:
                print(f"{result['name']}: {result['score']}/{result['max_score']}")

    return {
        "score": total_score,
        "tests": results
    }


# ======================================================
# Entry
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradescope", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=120,
                        help="max seconds per test")
    args = parser.parse_args()

    try:
        output = main(args)

        if args.gradescope:
            with open("results.json", "w") as f:
                json.dump(output, f)
        else:
            print("\nTOTAL:", output["score"])
            print(json.dumps(output, indent=2))

    except Exception:
        message = traceback.format_exc()
        print("Autograder crashed:\n", message)

        if args.gradescope:
            with open("results.json", "w") as f:
                json.dump(fail(message), f)
