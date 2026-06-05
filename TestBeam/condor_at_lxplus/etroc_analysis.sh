#!/bin/bash


# EDIT HERE:
# Change the run number in run_name to the number of the run you will analyze
# Change the run_input to the folder of the run you want to analyze
# Change the eos_output to a folder inside your /eos/user/u/username cernbox where you want to store the outputs. This can be the same for all runs.
# Make sure the run_name value is defined in the board_config file that you are using.
run_name="run18"
run_input="/eos/project/c/ctpps/PPS2/TestBeam/2026-06/ETROC-Data/run_018_beam_20260604_run_alignment_pos_h2p37_v4p37"
eos_output="etroc"
board_config="../board_configs_yaml/CERN_TB_H6_2026April_CELip.yaml"



# Don't edit below
#######################################

if [ "$1" = "" ]; then   
    echo ''   
    echo 'Usage: etroc_analysis.sh <stepNumber>'   
    echo ''   
    exit 1 
fi

step=$1
dryrun=""

if [[ -n "$2" ]]; then
    if [[ "$2" == "--dryrun" ]]; then
        dryrun="$2"
    else
        echo "Unknown option: $2"
        exit 1
    fi
fi

out=${eos_output}/${run_name}

username=$USER
user_letter="${username:0:1}"
user_eos="/eos/user/${user_letter}/${username}"


out_abs=${user_eos}/${out}

case "$step" in
    1|2|3)
        if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
            source envs/load_python39.sh
            echo "Env sourced. Analysis CodiMD/Readme starts at step 4!"
        else
            echo "Step 1 must be sourced, not simply executed! Usage: source etroc_analysis.sh 1"
        fi
        ;;
    4)
        python submit/submit_decoding.py -d ${run_input} -o ${out} --condor_tag ${run_name} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 5) after jobs are finished!"
        fi
        ;;
    5)
        hitsdir=${out_abs}/hits
        filecount=$(ls -1 "$hitsdir"/*.feather 2>/dev/null | wc -l)
        python utils/merge_feathers.py -d ${out}/hits -n ${filecount} ${dryrun}
        echo "Merged ${filecount} feather files. You can proceed to step 6)."
        ;;
    6)
        python core/path_finder.py -p ${out}/merged_hits --cal-label lip_${run_name} --track-label lip_${run_name} -c ${board_config} -r ${run_name} -s 15 -m 10
        cp lip_${run_name}*.csv ${out_abs}/
        echo "Path finding done, you can proceed to step 7)."
        ;;
    7)
        python utils/reduce_number_of_track_candidates.py -f ${out_abs}/lip_${run_name}_tracks.csv -m 50 --ntrk_table
        echo "You can proceed to step 8)."
        ;;
    8)
        python submit/submit_extract_events_by_path.py -d ${out_abs}/merged_hits -t lip_${run_name}_tracks.csv -o ${out}/pathSel_${run_name} -c ${board_config} -r ${run_name} --cal_table lip_${run_name}_cal_table.csv --condor_tag ${run_name} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 9) after jobs are finished!"
        fi
        ;;
    9)
        pdir=${out_abs}/pathSel_${run_name}
        filecount=$(ls -1 "$pdir"/*.parquet 2>/dev/null | wc -l)
        python core/reshape_event_to_track.py -d ${out}/pathSel_${run_name} -o ${out}/${run_name}_AfterCuts -c ${board_config} -r ${run_name} -p 2 -b ${filecount}
        echo "Track based selection done, you can proceed to step 10)."
        ;;
    10)
        python submit/submit_apply_tdc_cuts.py -d ${out}/${run_name}_AfterCuts -c ${board_config} -r ${run_name} --TOALower 0 --TOAUpper 800 --distance_factor 3.0 --condor_tag ${run_name} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 11) after jobs are finished!"
        fi
        ;;
    11)
        python core/count_path_nevts.py -d ${out}/${run_name}_AfterCuts -o lip_${run_name} --tag _lip_${run_name}
        echo "Events counted. You can proceed to step 12)."
        ;;
    12)
        python submit/submit_bootstrap.py -d ${out}/${run_name}_AfterCuts -o lip_${run_name} -n 200 --minimum_nevt 100 --iteration_limit 3000 --condor_tag ${run_name} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 13) after jobs are finished!"
        fi
        ;;
    13)
        python core/fit_bootstrap_results.py -d bootstrap_lip_${run_name}_group1/ -o lip_${run_name} --tag _lip_${run_name}
        cp -r lip_${run_name} ${out_abs}/
        echo "All done! Final output is in: ${out_abs}/lip_${run_name}"
        echo "Please check the results, then you can use step 14 to copy to the common eos."
        ;;
    14)
        if [ -d "/eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_name}" ]; then
            echo "The ${run_name} folder already exists in the common eos. Please verify, and copy by hand if needed."
        else
            mkdir -p "/eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_name}"
            cp ${out_abs}/lip_${run_name}/* /eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_name}/
            echo "Analysis results copied to /eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_name}"
        fi
        ;;
    *)
        echo "Unknown step, exiting"
        ;;
esac