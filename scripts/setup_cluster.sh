#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
mkdir -p /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/checkpoints
ln -sfn /ceph/csedu-scratch/other/"$USER"/safety-assessment-av/checkpoints "$SCRIPT_DIR"/../checkpoints

mkdir -p /ceph/csedu-scratch/other//users/"$USER"/safety-assessment-av/lightning_logs
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
echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN99 ###"
./setup_venv.sh

# # make sure that there's also a virtual environment
# # on the GPU nodes
# echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
# ssh cn47 "
#   source .profile
#   cd $PWD;
#   ./setup_venv.sh;
# "

# echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN48 ###"
# ssh cn48 "
#   source .profile
#   cd $PWD;
#   ./setup_venv.sh;
