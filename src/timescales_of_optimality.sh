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
# This script runs a series of possible combinations of timescales of          #
# adjustment for various process simulated by the TractLSM, at EucFACE's six   #
# "ring" plots.                                                                #  
#                                                                              #
# N.B.: if running in windows, applying the "sed -i 's/\r$//' filename"        #
#       command on the file might be necessary before it can be made an        #
#       executable.                                                            #
#                                                                              #
# __author__ = "Manon E. B. Sabot"                                             #
# __version__ = "1.0 (02.11.2021)"                                             #
# __email__ = "m.e.b.sabot@gmail.com"                                          #
#                                                                              #
################################################################################


#######################################
# Main: generates runfiles, runs model
# Globals:
#   lots
# Arguments:
#   None
# Returns:
#   input, output, ()& log files)
#######################################

main(){

# run file generator
mkrunfile="${this_dir}${sl}preparation_scripts${sl}create_setup_file.sh"

# model run launcher
runmodel="${this_dir}${sl}ExecTractLSM"

get_undeclared_variables # retrieve default variables if undeclared
process_ids=() # track job ids to check for successful completion

# loop over the hydraulic legacies
for fH in "None" "14" "30" "60" "90" "180"; do

  # loop over the leaf nitrogen opt
  for fN in "None" "7" "14" "21" "28" "35"; do

    # loop over the rings
    for ring in "R1" "R2" "R3" "R4" "R5" "R6"; do

      ID=$(echo ${RANDOM:0:6})
      ${mkrunfile} -s EucFACE${ring} -m ProfitMax -L ${fH} -O ${fN} \
                   -R timescales \
                   -o "${this_dir}${sl}tmp${sl}irun${ring}${ID}.txt"
      ${runmodel} "${this_dir}${sl}tmp${sl}irun${ring}${ID}.txt" &
      process_ids+=("$!") # store job ids
      sleep 40s # let the job be registered
      check_runstatus # check whether too many processes are ongoing

    done

  done

done

# remove the run files
wait
tidy

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
usage: $0 runs sensitivity experiments for various root distributions.
The option (-c) is not mandatory.

OPTIONS:
   -h      show this message
   -c      number of CPUs for parallelisation
EOF

}


#######################################
# Retrieve undeclared variables
# Globals:
#   NCPUs
# Arguments:
#   None
# Returns:
#   NCPUs
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

  # check whether processes are finished
  process_ids_check=()

  # check whether the processes are all running
  for pid in ${process_ids[@]}; do

    # this line might have to change on another system
    if [[ $(ps p ${pid} |wc -l) > 1 ]]; then

      process_ids_check+=("${pid}")

    else

      :  # do nothing, the process is finished

    fi

  done

  if [[ (( "${#process_ids_check[@]}" -lt "${#process_ids[@]}" )) ]]; then

    process_ids=("${process_ids_check[@]}")

  fi

  if [[ (( "${#process_ids[@]}" -ge "${NCPUs}" )) ]]; then

    sleep 180s # wait before re-checking

  fi

done

}


#######################################
# Tidy unnecessary created files
# Globals:
#   this_dir, sl
# Arguments:
#   None
# Returns:
#   None
#######################################

tidy(){

rm -r "${this_dir}${sl}tmp"

}


#### execute functions #########################################################

# filter options that exist from those that don't
while getopts "hc:" OPTION; do

  case ${OPTION} in

    h)
      usage
      exit 1
      ;;

    c)
      NCPUs=${OPTARG}
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

# log file
log_file="${this_dir}${sl}tmp${sl}log.o$(date +%s)"

if [[ ! -d ${this_dir}${sl}tmp${sl} ]]; then

  mkdir ${this_dir}${sl}tmp

fi

# execute main & output run time in log
{ time main ; } >> ${log_file} 2>&1
