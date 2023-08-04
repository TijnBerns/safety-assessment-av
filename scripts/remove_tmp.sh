#! /usr/bin/env bash

rm -rfv /scratch/$USER/tmp
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rm -rfv /scratch/$USER/tmp
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rm -rfv /scratch/$USER/tmp
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rm -rfv /scratch/$USER/tmp
