"""
Automated submission checker for ML Challenge 2026.
Validates matching_results.tsv and candidate_pairs.tsv against official rules.
"""

import os
import sys
import argparse

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Import the official validator
sys.path.append(os.path.abspath("student_resource/utils"))
try:
    from validate_submission import validate
except ImportError:
    # Fallback if path is different
    from utils.validate_submission import validate


def check_submission_files(
    matching_path: str = "output/matching_results.tsv",
    candidate_path: str = "output/candidate_pairs.tsv",
    test_dir: str = "student_resource/dataset/test",
    check_ids: bool = False,
) -> bool:
    """
    Run full submission validation.

    Returns True if passed, False otherwise.
    """
    print(f"=== CHECKING SUBMISSION ===")
    print(f"Matching file : {matching_path}")
    print(f"Candidate file: {candidate_path}")
    print(f"Test directory: {test_dir}")

    if not os.path.exists(matching_path):
        print(f"ERROR: Matching file does not exist: {matching_path}")
        return False

    errors, warnings = validate(
        matching_path=matching_path,
        candidate_path=candidate_path if (candidate_path and os.path.exists(candidate_path)) else None,
        test_dir=test_dir,
        check_ids=check_ids,
    )

    for w in warnings:
        print(f"  [WARNING] {w}")

    if errors:
        print(f"\n[FAIL] {len(errors)} error(s) found:")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        return False

    print("\n[PASS] Submission files passed all formatting and integrity checks!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="student_resource/dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    success = check_submission_files(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
        check_ids=args.check_ids,
    )
    sys.exit(0 if success else 1)
