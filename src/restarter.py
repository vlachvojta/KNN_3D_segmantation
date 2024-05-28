#!/usr/bin/env python3
import argparse
import datetime
from email.message import EmailMessage
import os
import time
import smtplib
import subprocess


def time_since_modification(filename):
    return time.time() - os.stat(filename).st_mtime


def screen_send(session_name, window_number, command):
    screen_prefix = f'screen -S {session_name} -p {window_number} -X stuff'.split()
    subprocess.run(screen_prefix + [command + '\n'])


class FileWatcher:
    def __init__(self, session_name, filename, patience):
        self.filename = filename
        self.patience = patience
        self.target_session = session_name
        self.restart_cmd = f'echo RESTARTING ; ./train.sh'

    def check(self):
        waiting_time = time_since_modification(self.filename)
        print(datetime.datetime.now().isoformat(), f'Time since last update: {waiting_time:.2f} s')
        if waiting_time > self.patience:
            self.restart()

    def restart(self):
        msg = EmailMessage()
        msg['Subject'] = 'pero-ocr on pcsevsik failed'
        msg['From'] = 'xvlach22@pcsevcik.fit.vutbr.cz'
        msg['To'] = 'xvlach22@stud.fit.vutbr.cz'

        msg.set_content(f'Since last update to {self.filename}, {time_since_modification(self.filename):.2f} s elapsed.')

        with smtplib.SMTP('kazi.fit.vutbr.cz') as s:
            s.send_message(msg)

        screen_send(self.target_session, 0, f'^C')
        time.sleep(5.0)
        screen_send(self.target_session, 0, self.restart_cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-patience', type=float, default=180.0)
    parser.add_argument('--check-interval', type=float, default=180.0)
    parser.add_argument('log_file')
    parser.add_argument('target_session')
    args = parser.parse_args()

    watcher = FileWatcher(args.target_session, args.log_file, args.max_patience)
    watcher.check()

    while True:
        time.sleep(args.check_interval)
        watcher.check()


if __name__ == '__main__':
    main()
