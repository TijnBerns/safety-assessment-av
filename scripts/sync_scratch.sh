#! /usr/bin/env bash

echo "### SYNCING CN77 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SYNCING CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SYNCING CN48 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rsync -a cn84:/scratch/tberns/ /scratch/tberns/
