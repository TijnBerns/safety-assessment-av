#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/ray_results/ray
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/ray_results/tmp/ray /tmp/ray


mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/ray_results
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/ray_results "$SCRIPT_DIR"/../ray_results

mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/lightning_logs
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/lightning_logs "$SCRIPT_DIR"/../lightning_logs

mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/logs
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/logs "$SCRIPT_DIR"/../logs

mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/estimates
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/estimates "$SCRIPT_DIR"/../estimates

# # place `~/.cache`, and optionally `~/.local`, in the ceph user directory in order to
# # save disk space in $HOME folder
# function setup_link {
#   dest_path=$1
#   link_path=$2

#   if [ -L "${link_path}" ] ; then
#     # link_path exists as a link
#     if [ -e "${link_path}" ] ; then
#       # and works
#       echo "link at $link_path is already setup"
#     else
#       # but is broken
#       echo "link $link_path is broken... Does $dest_path exists?"
#       return 1
#     fi
#   elif [ -e "${link_path}" ] ; then
#     # link_path exists, but is not a link
#     mkdir -p "$dest_path"
#     echo "moving all data in $link_path to $dest_path"
#     mv "$link_path"/* "$dest_path"/
#     rmdir "$link_path"
#     ln -s "$dest_path" "$link_path"
#     echo "created link $link_path to $dest_path"
#   else
#     # link_path does not exist
#     mkdir -p "$dest_path"
#     ln -s "$dest_path" "$link_path"

#     echo "created link $link_path to $dest_path"
#   fi

#   return 0
# }

# set up a virtual environment
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN84 ###"
./setup_venv.sh

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN77 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn77 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rsync -a cn84:/scratch/tberns/ /scratch/tberns/

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rsync -a cn84:/scratch/tberns/ /scratch/tberns/
