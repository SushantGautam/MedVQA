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
    parser.add_argument('competition', type=str,
                        help='Name of the competition (e.g., gi-2025)')
    parser.add_argument('task', type=str, help='Task number (1 or 2)')
    parser.add_argument('submission_repo', type=str,
                        help='Path to the submission repository')

    args = parser.parse_args()
    print("Running with arguments:", args)

    # Check if competition directory exists
    competition_dir = os.path.join(
        '/Users/sgautam/Documents/MedVQA', args.competition)
    if not os.path.isdir(competition_dir):
        raise FileNotFoundError(
            f"Competition '{args.competition}' does not exist! Need to update library?"+report)

    # Check if task file exists
    task_file = os.path.join(competition_dir, f'task{args.task}', 'run.py')
    if not os.path.isfile(task_file):
        raise FileNotFoundError(f"Task '{args.task}' does not exist!"+report)
    script_path = f'medvqa/{args.competition}/task{args.task}/run.py'
    subprocess.run(['python', script_path, args.submission_repo])


if __name__ == '__main__':
    main()
