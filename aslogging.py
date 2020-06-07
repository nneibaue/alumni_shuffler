'''Utils for logging. This is named aslogging so as not to conflict with logging from std library'''

import os
from datetime import datetime

def get_logfile():
  if not os.path.isdir('logs'):
    os.makedirs('logs')
  today=datetime.now().strftime('%Y-%m-%d')
  log_fname = f'alumni_shuffler_log_{today}.txt'
  return log_fname

def log(s):
  fname = get_logfile()
  timestamp = datetime.now().strftime('%H:%M:%S')
  with open(f'logs/{fname}', 'a') as f:
    f.write(f'\n{timestamp} >>>{s}')
