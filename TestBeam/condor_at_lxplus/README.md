## How to run this analysis

### 1. Load python 3.9 enviornment
```
source envs/load_python39.sh
```

### 2. Locate binary files in your EOS
Copy from DAQ PC to CernBox. Use `scp`, `rsync`, `xrdcp`.

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
- `-s` option: determine the fraction of data to read from each file.
- `-m` opiton: minimum threshold for track candidates occurence.
- `--cal_table_only` option: if this argument is included, the script will stop right after finding the cal code table.
- `--exclude_role` option: if this argument is included, the script will discard the board in the path finding. Possible argument: `trig, dut, ref, extra`

### 7. Check path finding output
```
python utils/reduce_number_of_track_candidates.py -f <FILE> -m <NUMBER> [--ntrk_table]
```
- `-f` option: path to "track-label" file from step 6.
- `-m` option: "new" minimum threshold for track candidates occurence.
- `--ntrk_table` option: if this argument is included, the script will show the table with the number of paths that survived from pre-defined thresholds.

### 8. Submit jobs for event selection by path
```
python submit/submit_extract_events_by_path.py -d <DIRNAME> -t <TRACK> -c <CONFIG> -r <RUNNAME> --cal_table <CAL_TABLE> -o <OUTNAME> --condor_tag <CONDOR_TAG> [--dryrun]
```

### 9. Reshaping the output from event-based to track-based
```
python core/reshape_to_tracks.py -d <DIRNAME> -o <OUTDIR> -r <RUNNAME> -c <CONFIG> --groups <GROUPS>
```

### 10. Submit jobs for apply TDC cuts
```
python submit/submit_apply_tdc_cuts.py -d <INPUTDIR> -c <CONFIG> -r <RUNNAME> --TOALower <TOALOWER> --TOAUpper <TOAUPPER> --dutTOTlower <DUTTOTLOWER> --dutTOTupper <DUTTOTUPPER> --distance_factor <DISTANCE_FACTOR> --condor_tag <CONDOR_TAG> [--dryrun]
```

### 11. Count the number of events per track
```
python core/count_path_nevts.py -d <INPUTDIR> -o <OUTPUTDIR> [--tag <TAG>]
```
- `-d` option: path to the directory including the step 10 outputs.
- `-o` option: path to the output directory.
- `--tag` option: addtional string if needed.

### 12. Submit jobs for bootstrap
```
python submit/submit_bootstrap.py -d <DIRNAME> -o <OUTPUTDIR> -n <NUM_BOOTSTRAP_OUTPUT> --minimum_nevt <MINIMUM_NEVT> --iteration_limit <ITERATION_LIMIT> --condor_tag <CONDOR_TAG> [--dryrun]
```

### 13. Merge bootstrap results (unbinned gaussian fit)
```
python core/merge_bootstrap_results.py -d <INPUTDIR> -o <OUTPUTDIR> --minimum <MINIMUM> [--tag <TAG>] [--hist_bins <HIST_BINS>]
```
- `-d` option: path to the directory including boostrap output files.
- `-o` option: path to the output directory. **Recommend to use the same output directory in step 11.**
- `--minimum` option: the threshold for number of bootstrap results. If number of results is less than the threshold, the path is ignored for the final result.
- `--tag` option: addtional string if needed.
- `--hist_bins` option: number of histogram for binned fit. If unbinned fit is failed.