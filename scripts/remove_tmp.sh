#! /usr/bin/env bash

srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rm -rfv /scratch/tberns/tmp
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rm -rfv /scratch/tberns/tmp
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rm -rfv /scratch/tberns/tmp
