#!/usr/bin/env bash

################################################################################
#                                                                              #
# This file is part of the TractLSM project.                                   #
#                                                                              #
# Copyright (c) 2022 Manon E. B. Sabot                                         #
#                                                                              #
# Please refer to the terms of the MIT License, which you should have received #
# along with this script.                                                      #
#                                                                              #
# This script calculates and calibrates necessary model parameters, using as   #
# much observational data from EucFACE's six "ring" plots as possible.         #
#                                                                              #
# N.B.: if running in windows, applying the "sed -i 's/\r$//' filename"        #
#       command on the file might be necessary before it can be made an        #
#       executable.                                                            #
#                                                                              #
# __author__ = "Manon E. B. Sabot"                                             #
# __version__ = "1.0 (12.10.2021)"                                             #
# __email__ = "m.e.b.sabot@gmail.com"                                          #
#                                                                              #
################################################################################


#######################################
# Main: calculates and calibrates
#       model parameters
# Globals:
#   lots
# Arguments:
#   None
# Returns:
#   parameter files
#######################################

main(){

# calc params that can directly be calculated
${this_dir}${sl}preparation${sl}calc_model_params.py
wait

# ensure the soil water files exist
${this_dir}${sl}preparation${sl}soil_moisture_profiles.py
wait

get_undeclared_variables # retrieve default variables if undeclared
process_ids=() # track job ids to check for successful completion

# calibrate the model parameter
for process in "intercept" "trans" "evap"; do

  for ring in "R1" "R2" "R3" "R4" "R5" "R6"; do

    if  [[ (( "${process}" == "trans" )) ]]; then

      for model in "ProfitMax" "Medlyn"; do

        ${this_dir}${sl}preparation${sl}calib_model_params.py -p ${process} \
                                                              -w ${ring} \
                                                              -m ${model} \
                                                              -R ${project} &
        process_ids+=("$!") # store job ids
        sleep 40s # let the job be registered
        check_runstatus # check whether too many processes are ongoing

      done

    else

      ${this_dir}${sl}preparation${sl}calib_model_params.py -p ${process} \
                                                            -w ${ring} \
                                                            -R ${project} &
      process_ids+=("$!") # store subjob ids
      sleep 40s # let the job be registered
      check_runstatus # check whether too many processes are ongoing

    fi

  done

done
wait

# put info into parameter file once jobs are finished
${this_dir}${sl}preparation${sl}populate_param_file.py ${fparams}
wait

if [[ ${project} == "tmp_calib" ]]; then

  tidy # clean up temporary driving files used for calibration

fi

}


#### other functions are defined in this section ###############################

#######################################
# Usage descprition for this script
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   Prints usage message in shell
#######################################

usage(){ # user manual on how to use this script

cat << EOF
usage: $0 computes necessary model parameters.
None of the options (-c, -R, & -p) are mandatory.

OPTIONS:
   -h      show this message
   -c      number of CPUs for parallelisation
   -R      project repository of the (temporary) calibration drivers
   -p      parameter file in which to put the parameter estimates
EOF

}


#######################################
# Retrieve undeclared variables
# Globals:
#   NCPUs, project, fparams
# Arguments:
#   None
# Returns:
#   NCPUs, project
#######################################

get_undeclared_variables(){

# NCPUs for parallelisation
if [[ -z ${NCPUs+x} ]]; then

  unicore=$(grep ^cpu\\scores /proc/cpuinfo |uniq |awk '{print $4}')

  # check whether threads are physical or not
  if [[ (( $(grep -E "cpu cores|siblings|physical id" /proc/cpuinfo \
             |xargs -n 11 echo |sort |uniq |awk '/^physical id/{print $7}' \
             |tail -1) -le $(grep -E "cpu cores|siblings|physical id" \
             /proc/cpuinfo |xargs -n 11 echo |sort |uniq \
             |awk '/^physical id/{print $11}' |tail -1) )) && \
        (( ${unicore} -lt $(nproc --all) )) ]]; then

    NCPUs=${unicore}

  else

    NCPUs=$((${unicore} / 2)) # hyperthreading, take half the CPUs

  fi
fi

# repo for temporary storage of the calibration files
if [[ -z ${project+x} ]]; then

  project="tmp_calib"

fi

# parameter file for storage of the parameter estimates
if [[ -z ${fparams+x} ]]; then

  fparams="${data_dir}${sl}input${sl}site_params.csv"

fi

}


#######################################
# Check status of subprocesses
# Globals:
#   process_ids, NCPUs
# Arguments:
#   None
# Returns:
#   smaller process_id array
#######################################

check_runstatus(){

while [[ (( "${#process_ids[@]}" -ge "${NCPUs}" )) ]]; do

  i=0 # number tracker

  # check whether the processes are all running
  for pid in ${process_ids[@]}; do

    # this line might have to change on another system
    if [[ $(ps p ${pid} | wc -l) > 1 ]]; then

      : # do nothing, it's running

    else

      unset process_ids[i] # remove finished process id

    fi

    let i+=1 # update number tracker

  done

  # are there still too many processes on?
  if [[ (( "${#process_ids[@]}" -ge "${NCPUs}" )) ]]; then

    sleep 120s # wait before re-checking

  fi

done

}


#######################################
# Tidy unnecessary created files
# Globals:
#   data_dir, sl, project
# Arguments:
#   None
# Returns:
#   None
#######################################

tidy(){

rm -r "${data_dir}${sl}input${sl}projects${sl}${project}"

}


#### execute functions #########################################################

# filter options that exist from those that don't
while getopts "hc:R:p:" OPTION; do

  case ${OPTION} in

    h)
      usage
      exit 1
      ;;

    c)
      NCPUs=${OPTARG}
      ;;

    R)
      project=${OPTARG}
      ;;

    p)
      fparams=${OPTARG}
      ;;

    ?)
      usage
      exit 1
      ;;

  esac

done


# standard sanity checks and file locations
this_dir="$(cd "${BASH_SOURCE%/*}"; pwd)"

# deal with linux & windows compatibility
if [[ ${this_dir} == *\/* ]]; then

  sl="/"

elif [[ ${this_dir} == *\\* ]]; then

  sl="\\"

else

  echo "Unknown dir path format, now exiting."
  exit 1

fi

# make sure this is located above the TractLSM dir...
if [[ ${this_dir} == *"TractLSM"* ]]; then

  this_dir=$(echo ${this_dir} |awk -F ${sl}"TractLSM" '{print $1}' |xargs)

fi

if [[ ${this_dir} == *"src"* ]]; then

  data_dir=$(echo ${this_dir} |awk -F ${sl}"src" '{print $1}' |xargs)

else

  data_dir=${this_dir}

fi

# log file
log_file="${this_dir}${sl}tmp${sl}log.o$(date +%s)"

if [[ ! -d ${this_dir}${sl}tmp${sl} ]]; then

  mkdir ${this_dir}${sl}tmp

fi

# execute main & output run time in log
{ time main ; } >> ${log_file} 2>&1
