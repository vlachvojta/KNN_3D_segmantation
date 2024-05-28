#!/usr/bin/env python3
import argparse
import datetime
import time
import subprocess


def start_process(session_name, cmd, log_file):
    message = f'\nStarting process {cmd} in session {session_name} at {datetime.datetime.now()}'
    with open(log_file, 'a') as f:
        f.write(message)

    print(message)
    subprocess.run(['tmux', 'new', '-d', '-s', session_name, cmd])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=60*30, help='Interval between checks in seconds (default: 30 minutes)')
    parser.add_argument('--log_file', type=str, default='restarter.log', help='Log file to write to (default: restarter.log)')
    parser.add_argument('--target_session', type=str, default='KNN_training', help='Name of the tmux session to monitor (default: KNN_training)')
    parser.add_argument('--cmd', type=str, default='./train.sh', help='Command to run in the tmux session (default: ./train.sh)')
    args = parser.parse_args()

    while True:
        start_process(args.target_session, args.cmd, args.log_file)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
