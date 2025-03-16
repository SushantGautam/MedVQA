import argparse
import subprocess
import os


report = '''\n⚠️⚠️⚠️\n
Try installing latest version of the library by running the following command:
\n pip install git+https://github.com/SushantGautam/MedVQA.git
\n If you cannot solve the problem add an issue at https://github.com/SushantGautam/MedVQA/issues and report the log above! We will try to solve the problem as soon as possible.\n
⚠️⚠️⚠️'''


def main():
    print("MedVQA CLI")
    parser = argparse.ArgumentParser(description='MedVQA CLI')
    parser.add_argument('--competition', type=str, required=True,
                        help='Name of the competition (e.g., gi-2025)')
    parser.add_argument('--task', type=str, required=True,
                        help='Task number (1 or 2)')
    parser.add_argument('--submission_repo', type=str, required=True,
                        help='Path to the submission repository')

    args = parser.parse_args()

    # Dynamically find the base directory of the MedVQA library
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if competition directory exists
    competition_dir = os.path.join(base_dir, 'competitions', args.competition)
    if not os.path.isdir(competition_dir):
        raise FileNotFoundError(
            f"Competition '{args.competition}' does not exist at {competition_dir}! Need to update library?"+report)
    # Check if task file exists
    task_file = os.path.join(competition_dir, f'task_{args.task}', 'run.py')
    if not os.path.isfile(task_file):
        raise FileNotFoundError(
            f"Task '{args.task}' does not exist at {task_file}!"+report)
    subprocess.run(['python', task_file, args.submission_repo])


if __name__ == '__main__':
    main()
