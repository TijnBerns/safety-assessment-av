#! /usr/bin/env bash

temp_folder = /scratch/tberns/tmp

echo "### REMOVING $temp_folder FROM CN77 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rm -rfv $temp_folder

echo "### REMOVING $temp_folder FROM CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rm -rfv $temp_folder

echo "### REMOVING $temp_folder FROM CN48 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rm -rfv $temp_folder
