import sys
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Task 1 specific arguments')
    parser.add_argument('--example_arg', type=str,
                        help='An example argument for Task 1')
    return parser


def main(submission_repo, example_arg):
    print(
        f"Running GI-2025 Task 1 with repository: {submission_repo} and example_arg: {example_arg}")


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.submission_repo, args.example_arg)
