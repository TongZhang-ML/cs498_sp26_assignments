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
import importlib
import os


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
            output="failed"
        )


# ======================================================
# Main
# ======================================================

def load_suite(module_name):
    """
    Lazy import of test module.
    Prevents crashes if student file is missing.
    """
    module = importlib.import_module(module_name)
    return module.ALL_TESTS


def main(args):

    suite_specs = [
        ("GRU", "tests_gru", "gru.py"),
        ("VAE", "tests_vae", "VAE.py"),
    ]

    results = []
    total_score = 0

    for suite_name, test_module_name, required_file in suite_specs:

        # --------------------------------------------------
        # Check submission file FIRST
        # --------------------------------------------------
        if not os.path.isfile(required_file):

            msg = (
                f"Missing required submission file: {required_file}\n"
                f"Please submit {required_file} to run the {suite_name} tests."
            )

            # We still want correct max score → load tests only to count points
            try:
                suite = load_suite(test_module_name)
                max_points = sum(points for _, points, _ in suite)
            except Exception:
                max_points = 0

            results.append(
                make_result(
                    name=f"{suite_name} :: FILE MISSING",
                    score=0,
                    max_score=max_points,
                    output=msg
                )
            )

            if not args.gradescope:
                print(f"{suite_name}: SKIPPED (missing {required_file})")

            continue

        # --------------------------------------------------
        # Now safe to import tests
        # --------------------------------------------------
        suite = load_suite(test_module_name)

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
        print(output)
        if args.gradescope:
            with open("results.json", "w") as f:
                json.dump(output, f)
        else:
            print("\nTOTAL:", output["score"])
            print(json.dumps(output, indent=2))

    except Exception:
        message = traceback.format_exc()
        print("Autograder crashed:\n", message)
        print(message)
        if args.gradescope:
            with open("results.json", "w") as f:
                json.dump(fail(message), f)
