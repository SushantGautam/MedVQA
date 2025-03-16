import argparse
import subprocess

def main():
    print("MedVQA CLI")
    parser = argparse.ArgumentParser(description='MedVQA CLI')
    parser.add_argument('competition', type=str, help='Name of the competition (e.g., gi-2025)')
    parser.add_argument('task', type=int, choices=[1, 2], help='Task number (1 or 2)')
    parser.add_argument('submission_repo', type=str, help='Path to the submission repository')

    args = parser.parse_args()

    script_path = f'medvqa/{args.competition}/task{args.task}/run.py'

    subprocess.run(['python', script_path, args.submission_repo])

if __name__ == '__main__':
    main()
