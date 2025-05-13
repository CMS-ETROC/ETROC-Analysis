#!/usr/bin/env python

import fileinput
import re
from natsort import natsorted
from datetime import datetime
from tabulate import tabulate

clusterID = None
job_list = {}

cluster_re = re.compile(r"^ClusterID\s+=\s+(\d+)")
process_re = re.compile(r"^ProcID\s+=\s+(\d+)")
toe_re = re.compile(r"^ToE\s+=\s+\[(.+)\]")

exit_re = re.compile(r"ExitBySignal\s+=\s+([a-zA-Z]+);")
how_code_re = re.compile(r"HowCode\s+=\s+(\d+);")
when_re = re.compile(r"When\s+=\s+(\d+);")
who_re = re.compile(r'Who\s+=\s+\"?([a-zA-Z]+)\"?;')
exit_code_re = re.compile(r"ExitCode\s+=\s+(\d+);")
how_re = re.compile(r'How\s+=\s+\"?([a-zA-Z_]+)\"?')

current_cluster = None
current_process = None
current_toe = {}

for line in fileinput.input():
    cluster_match = cluster_re.match(line)
    process_match = process_re.match(line)
    toe_match = toe_re.match(line)

    if cluster_match is not None:
      if current_cluster is not None:
        print("There is an issue, a job with multiple cluster IDs is present")
      current_cluster = cluster_match.group(1)
    elif process_match is not None:
      if current_process is not None:
        print("There is an issue, a job with multiple job IDs is present")
      current_process = process_match.group(1)
    elif toe_match is not None:
      toe_info = toe_match.group(1)

      #print(toe_info)

      exit_match = exit_re.search(toe_info)
      how_code_match = how_code_re.search(toe_info)
      when_match = when_re.search(toe_info)
      who_match = who_re.search(toe_info)
      exit_code_match = exit_code_re.search(toe_info)
      how_match = how_re.search(toe_info)

      if len(current_toe) > 0:
        print("there is an issue, a job with multiple ToEs is present")

      current_toe = {
        'exit': exit_match.group(1),
        'exit_code': exit_code_match.group(1),
        'how': how_match.group(1),
        'how_code': how_code_match.group(1),
        'who': who_match.group(1),
        'when': datetime.fromtimestamp(int(when_match.group(1))),
      }
    else: # This would be safer if I look for the empty line, something to improve in the future
      # End of info, save and reset
      if current_cluster is not None and current_process is not None and len(current_toe) > 0:
        if f"{current_cluster}.{current_process}" in job_list:
          print("There is an issue, the same job shows up multiple times")
        else:
          job_list[f"{current_cluster}:{current_process}"] = current_toe

      current_cluster = None
      current_process = None
      current_toe = {}

#print(job_list)
print(
  tabulate(
    [[x, job_list[x]['exit_code'], job_list[x]['who'], job_list[x]['how'], job_list[x]['when']] for x in natsorted(job_list.keys())],
    headers=['Job Unique ID', 'Exit Code', 'Who', 'How', 'When']
  )
)
