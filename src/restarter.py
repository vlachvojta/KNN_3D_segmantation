#!/usr/bin/env python3
import argparse
import datetime
import time
import subprocess

def log_stuff(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
    print(message)

def start_process(session_name, cmd, log_file):
    log_stuff(log_file, f'\nStarting process {cmd} in session {session_name} at {datetime.datetime.now()}')
    subprocess.run(['tmux', 'new', '-d', '-s', session_name, cmd])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=60*30, help='Interval between checks in SECONDS (default: 30 minutes)')
    parser.add_argument('--log_file', type=str, default='restarter.log', help='Log file to write to (default: restarter.log)')
    parser.add_argument('--target_session', type=str, default='KNN_training', help='Name of the tmux session to monitor (default: KNN_training)')
    parser.add_argument('--cmd', type=str, default='./train.sh', help='Command to run in the tmux session (default: ./train.sh)')
    parser.add_argument('--wait_before_start', type=float, default=0, help='Interval between checks in SECONDS (default: 0)')
    args = parser.parse_args()

    log_stuff(args.log_file, f'Starting restarter with interval {args.interval} and target session {args.target_session} at {datetime.datetime.now()}.')
    log_stuff(args.log_file, f'Will start after {args.wait_before_start} seconds ({args.wait_before_start // 60} minutes).')

    time.sleep(args.wait_before_start) # Schedule start time

    while True:
        start_process(args.target_session, args.cmd, args.log_file)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
