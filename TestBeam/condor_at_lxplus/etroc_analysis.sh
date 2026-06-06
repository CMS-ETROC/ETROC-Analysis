#!/bin/bash


# EDIT HERE:
# Change the run number in run_name to the number of the run you will analyze
# Change the run_input to the folder of the run you want to analyze
# Change the output_folder_in_eos to a folder inside your /eos/user/u/username cernbox where you want to store the outputs. This can be the same for all runs.
# Make sure the run_name value is defined in the board_config file that you are using.
run_name="run29"
analysis_label="beam_20260606_PhysicsRunDUT220V_180V_All-config1"
run_input="/eos/project/c/ctpps/PPS2/TestBeam/2026-06/ETROC-Data/run_029_beam_20260606_PhysicsRunDUT220V_180V_All"
output_folder_in_eos="etroc"
board_config="../board_configs_yaml/CERN_TB_H6_2026April_CELip.yaml"



# Don't edit below
#######################################


force=false
dryrun=""
args=()

# get optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f)
            force=true
            ;;
        --dryrun)
            dryrun="--dryrun"
            ;;
        *)
            args+=("$1")
            ;;
    esac
    shift
done

# Restore positional args
set -- "${args[@]}"

step=$1



is_empty_dir() {
    local dir=$1

    [[ ! -d "$dir" ]] && return 0

    [[ -z $(find "$dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null) ]]
}




# FIX THIS. Step < 4: SOURCE the script. Step > 3: do NOT source the script. Exit safely, also handle if step is empty
if [ "$step" = "" ]; then   
    echo ''   
    echo 'Usage: etroc_analysis.sh <stepNumber> [-f] [--dryrun]'   
    echo ''   
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return 1
    else
        exit 1
    fi
fi


if [ "$step" -gt 3 ]; then
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        echo "Do not source the script from step 4! Just execute, usage: ./etroc_analysis.sh <step> [-f] [--dryrun]"
        return 1
    fi
fi



run_label=${run_name}-${analysis_label}

out=${output_folder_in_eos}/${run_label}

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
            exit 1
        fi

        if ! is_empty_dir "${out_abs}" && ! $force; then
            echo "Warning: The specified output directory ${out_abs} is NOT EMPTY!"
        fi
        ;;
    4)

        if ! is_empty_dir "${out_abs}" && ! $force; then
            echo "Error: The specified output directory ${out_abs} is NOT EMPTY! Run the script with -f if you want to overwrite."
            exit 1
        fi

        python submit/submit_decoding.py -d ${run_input} -o ${out} --condor_tag ${run_label} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 5) (optional) or 6) after jobs are finished!"
        fi
        ;;
    5)
        if is_empty_dir ${out_abs}/hits; then
            echo "Error: The output of step 4) ${out_abs}/hits is EMPTY! Cannot proceed."
            exit 1
        fi

        if ! is_empty_dir "${out_abs}"/merged_hits && ! $force; then
            echo "Error: The specified output directory ${out_abs}/merged_hits is NOT EMPTY! Run the script with -f if you want to overwrite."
            exit 1
        fi

        hitsdir=${out_abs}/hits
        filecount=$(ls -1 "$hitsdir"/*.feather 2>/dev/null | wc -l)
        python utils/merge_feathers.py -d ${out}/hits -n ${filecount} ${dryrun}

        if [[ "$dryrun" == "" ]]; then
            if is_empty_dir "${out_abs}"/merged_hits; then
                echo "Error: merged_hits output is empty. Check messages for errors."
            else 
                echo "Merged ${filecount} feather files. You can proceed to step 6)."
            fi
        fi
        ;;
    6)
        hitsdir=${out}/hits    # needs to be eos-relative
        if ! is_empty_dir "${out_abs}/merged_hits"; then
            hitsdir="${out}/merged_hits"
        elif is_empty_dir "${out_abs}/hits"; then
            echo "Error: The output of step 4) ${out_abs}/hits is EMPTY! Cannot proceed."
            exit 1
        fi

        if [[ -f "lip_${run_label}_tracks.csv" ]] && ! $force; then
            echo "Error: lip_${run_label}_tracks.csv already exists.  Run the script with -f if you want to overwrite."
            exit 1
        fi
        if [[ -f "lip_${run_label}_cal_table.csv" ]] && ! $force; then
            echo "Error: lip_${run_label}_cal_table.csv already exists.  Run the script with -f if you want to overwrite."
            exit 1
        fi

        python core/path_finder.py -p ${hitsdir} --cal-label lip_${run_label} --track-label lip_${run_label} -c ${board_config} -r ${run_name} -s 15 -m 10
        cp lip_${run_label}*.csv ${out_abs}/

        if [[ ! -f "lip_${run_label}_tracks.csv" ]] || [[ ! -f "lip_${run_label}_cal_table.csv" ]]; then
            echo "Some output files were not created. Check for errors and try again."
            exit 1
        else
            echo "Path finding done, you can proceed to step 7)."
        fi

        ;;
    7)
        if [[ ! -f "lip_${run_label}_tracks.csv" ]]; then
            echo "Error: lip_${run_label}_tracks.csv from step 6) does not exist. Cannot proceed."
            exit 1
        fi

        python utils/reduce_number_of_track_candidates.py -f lip_${run_label}_tracks.csv -m 50 --ntrk_table
        echo "You can proceed to step 8)."
        ;;
    8)

        hitsdir=${out_abs}/hits   # need sto be absolute
        if ! is_empty_dir "${out_abs}/merged_hits"; then
            hitsdir="${out_abs}/merged_hits"
        elif is_empty_dir "${out_abs}/hits"; then
            echo "Error: The output of step 4) ${out_abs}/hits is EMPTY! Cannot proceed."
            exit 1
        fi


        if [[ ! -f "lip_${run_label}_cal_table.csv" ]]; then
            echo "Error: lip_${run_label}_cal_table.csv from step 6) does not exist. Cannot proceed."
            exit 1
        fi


        if ! is_empty_dir "${out_abs}/pathSel_${run_label}" && ! $force; then
            echo "Error: The specified output directory ${out_abs}/pathSel_${run_label} is NOT EMPTY! Run the script with -f if you want to overwrite."
            exit 1
        fi


        python submit/submit_extract_events_by_path.py -d ${hitsdir} -t lip_${run_label}_tracks.csv -o ${out}/pathSel_${run_label} -c ${board_config} -r ${run_name} --cal_table lip_${run_label}_cal_table.csv --condor_tag ${run_label} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 9) after jobs are finished!"
        fi
        ;;
    9)
        if is_empty_dir "${out_abs}/pathSel_${run_label}"; then
            echo "Error: The output of step 8) ${out_abs}/pathSel_${run_label} is EMPTY! Cannot proceed."
            exit 1
        fi

        if ! is_empty_dir "${out_abs}/${run_label}_AfterCuts" && ! $force; then
            echo "Error: The specified output directory ${out_abs}/${run_label}_AfterCuts is NOT EMPTY! Run the script with -f if you want to overwrite."
            exit 1
        fi

        pdir=${out_abs}/pathSel_${run_label}
        filecount=$(ls -1 "$pdir"/*.parquet 2>/dev/null | wc -l)
        python core/reshape_event_to_track.py -d ${out}/pathSel_${run_label} -o ${out}/${run_label}_AfterCuts -c ${board_config} -r ${run_name} -p 1 -b ${filecount}

        if is_empty_dir "${out_abs}/${run_label}_AfterCuts" ; then
            echo "Output not created in ${out_abs}/${run_label}_AfterCuts, check messages for errors!"
        else 
            echo "Track based selection done, you can proceed to step 10)."
        fi
        ;;
    10)
        if is_empty_dir "${out_abs}/${run_label}_AfterCuts"; then
            echo "Error: The output of step 9) ${out_abs}/${run_label}_AfterCuts is EMPTY! Cannot proceed."
            exit 1
        fi


        python submit/submit_apply_tdc_cuts.py -d ${out}/${run_label}_AfterCuts -c ${board_config} -r ${run_name} --TOALower 0 --TOAUpper 800 --distance_factor 3.0 --condor_tag ${run_label} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 11) after jobs are finished!"
        fi
        ;;
    11)
        if is_empty_dir "${out_abs}/${run_label}_AfterCuts/time"; then
            echo "Error: The time output of step 10) in ${out_abs}/${run_label}_AfterCuts/time is EMPTY! Cannot proceed."
            exit 1
        fi

        if [[ -f "lip_${run_label}/nevt_lip_${run_label}.csv" ]]  && ! $force; then
            echo "Error: lip_${run_label}/nevt_lip_${run_label} from step 11) already exists.  Run the script with -f if you want to overwrite."
            exit 1
        fi

        python core/count_path_nevts.py -d ${out}/${run_label}_AfterCuts -o lip_${run_label} --tag _lip_${run_label}

        if [[ ! -f "lip_${run_label}/nevt_lip_${run_label}.csv" ]]; then
            echo "Output file was not created, check messages for errors."
        else
            echo "Events counted. You can proceed to step 12)."
        fi

        ;;
    12)
        if is_empty_dir "${out_abs}/${run_label}_AfterCuts"; then
            echo "Error: The output of step 9) ${out_abs}/${run_label}_AfterCuts is EMPTY! Cannot proceed."
            exit 1
        fi
        if is_empty_dir "${out_abs}/${run_label}_AfterCuts/time"; then
            echo "Error: The time output of step 10) in ${out_abs}/${run_label}_AfterCuts/time is EMPTY! Cannot proceed."
            exit 1
        fi

        if ! is_empty_dir "bootstrap_lip_${run_label}" && ! $force; then
            echo "Error: The specified output directory bootstrap_lip_${run_label} is NOT EMPTY! Run the script with -f if you want to overwrite."
            exit 1
        fi

        python submit/submit_bootstrap.py -d ${out}/${run_label}_AfterCuts -o lip_${run_label} -n 200 --minimum_nevt 100 --iteration_limit 3000 --condor_tag ${run_label} ${dryrun}
        if [[ "$dryrun" == "" ]]; then
            echo "Condor jobs submitted! Only proceed to step 13) after jobs are finished!"
        fi
        ;;
    13)
        if is_empty_dir "bootstrap_lip_${run_label}"; then
            echo "Error: The specified directory bootstrap_lip_${run_label} is empty. Cannot proceed"
            exit 1
        fi

        if [[ -f "lip_${run_label}/resolution_table_lip_${run_label}.csv" ]]  && ! $force; then
            echo "Error: lip_${run_label}/resolution_table_lip_${run_label} from step 13) already exists.  Run the script with -f if you want to overwrite."
            exit 1
        fi

        python core/fit_bootstrap_results.py -d bootstrap_lip_${run_label} -o lip_${run_label} --tag _lip_${run_label}

        if [[ ! -f "lip_${run_label}/resolution_table_lip_${run_label}.csv" ]]; then
            echo "Error: Resolution table not created! Check messages for errors."
            exit 1
        fi

        echo "Copying... (this might take some time)"
        cp -r bootstrap_lip_${run_label} ${out_abs}/
        cp -r lip_${run_label} ${out_abs}/
        echo "All done! Final output is in: ${out_abs}/lip_${run_label}"
        echo "Please check the results, then you can use step 14 to copy to the common eos."
        ;;
    14)
        if [ -d "/eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_label}" ]; then
            echo "The ${run_label} folder already exists in the common eos. Please verify, and copy by hand if needed."
        elif ! is_empty_dir "${out_abs}/lip_${run_label}"; then
            mkdir -p "/eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_label}"
            cp ${out_abs}/lip_${run_label}/* /eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_label}/
            echo "Analysis results copied to /eos/project/c/ctpps/PPS2/TestBeam/2026-06/analysis_out/${run_label}"
        else
            echo "Analysis for ${run_label} is not yet finished, aborting"
        fi
        ;;
    *)
        echo "Unknown step $step, exiting"
        ;;
esac