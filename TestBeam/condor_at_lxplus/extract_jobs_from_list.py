#!/usr/bin/env python

import fileinput
import pickle
import re
from tabulate import tabulate
from datetime import datetime

first_line = ""
found_first_line = False
first_line_index = None
current_line = 0

user = None
user_mismatch = False
job_list = {}

for line in fileinput.input():
  if not found_first_line:
    first_line_re = re.compile("^\s+ID\s+OWNER\s+SUBMITTED.+")
    match = first_line_re.match(line)
    if match is not None:
      found_first_line = True
      first_line = line
      first_line_index = current_line
      #print("Found first line")
  else:
    job_info_re = re.compile("^\s*(\d+.\d+)\s+([A-Za-z0-9]+)\s+(\d+/\d+\s+\d+:\d+).+")
    match = job_info_re.match(line)
    if match is not None:
      #print("Found a line with info:")
      #print(match.group(1,2,3))

      if user is None:
        user = match.group(2)
      elif user != match.group(2):
        user_mismatch = True
        user = match.group(2)

      submit_time = datetime.strptime(f"{datetime.now().year}/{match.group(3)}", '%Y/%m/%d %H:%M')

      unique_id = match.group(1)
      cluster_id = unique_id.split('.')[0]
      job_id = unique_id.split('.')[1]

      if cluster_id not in job_list:
        job_list[cluster_id] = {}
      if job_id in job_list[cluster_id]:
        print(f"There is an issue, the same job shows up twice in the list {unique_id}")
      else:
        job_list[cluster_id][job_id] = {
          'user': user,
          'submit': submit_time,
        }
    else:
      print("There is an unexpected issue, found a line which does not match the expected format")
  current_line += 1

if user_mismatch:
  print("There are multiple users in the list of jobs. Please double check how the list was created. Printing the job list anyway")

print(tabulate([[x, len(job_list[x]), min([job_list[x][y]['submit'] for y in job_list[x]])] for x in job_list], headers=['Cluster ID', 'Number of Jobs', 'First Submission']))

with open('my_job_list.pickle', 'wb') as f:
    pickle.dump(job_list, f, pickle.HIGHEST_PROTOCOL)
