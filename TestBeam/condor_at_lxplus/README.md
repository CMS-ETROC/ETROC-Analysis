## How to run this analysis

### 1. Load python 3.9 enviornment
```
source envs/load_python39.sh
```

### 2. Locate binary files in your EOS
### 3. Check condor server
```
condor_q
```
If you notice that the worker has too many jobs, you can switch the server by the following command:
```
myschedd bump
```

### 4. Submit decoding jobs
```
python submit/submit_decoding.py -d <absolute path to input directory> -o <relative path to output directry> --range N1 N2 --condor_tag <unique tag> --dryrun
```
- `-d` option: absolute path to the input directory. e.g. `/eos/user/n/name/directory`
- `-o` option: relative path to the output directory. The script knows the base path `/eos/user/n/name/`, so only need to specify the path after the base path.
- `--range N1 N2` option: Must be integers and `N1 < N2`. Determine the best number of files to merge in a given range.
- `--condor_tag` option: String to separate the job submission.
- `--dryrun` option: if this argument is included, actual submission will not happen. But still making the list, bash script, and condor jdl files.

### 5. (Optional) Merge output feather files
```
python utils/merge_feathers.py -d <INPUT_DIR> -n <NUMBER_OF_MERGE> --dryrun
```
- `-d` option: absolute path to the input directory.
- `-n` option: number of files for a single group.
- `--dryrun` option: if this argument is included, merge will not happen but showing the number of groups after merging.

### 6. Path finding
```
python core/path_finder.py -p <PATH> --cal-label <CAL_LABEL> --track-label <TRACK_LABEL> -c <CONFIG> -r <RUNNAME> -s <SAMPLING> -m <NTRACKS> [--cal_table_only] [--exclude_role <EXCLUDE_ROLE>]
```
- `-p` option: absolute path to the directory including feathers (outputs from step 4 or 5).
- `--cal-label` option: output name for cal table csv file.
- `--track-label` option: output name for track candidates csv file.
- `-c` option: path to the board config yaml file.
- `-r` option: "key" of dictionary in yaml file (usually run identifier).
- `-s` option: fraction of
- `-m` opiton: minimum threshold for track candidates occurence.
- `--cal_table_only` option: if this argument is included, the script will stop right after finding the cal code table.
- `--exclude_role` option: if this argument is included, the script will discard the board in the path finding. Possible argument: `trig, dut, ref, extra`