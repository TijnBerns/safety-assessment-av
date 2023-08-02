#! /usr/bin/env bash

echo "### SYNCING CN77 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SYNCING CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SYNCING CN48 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

# srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rsync -a /scratch/tberns/ cn84:/scratch/tberns/
# srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rm -rfv /scratch/tberns/ray_results 