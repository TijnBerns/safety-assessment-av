#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set variable to path of root of this project
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# We want to create the virtual environment in the scratch directory as /scratch/
# is a local disk (unique for each node) and therefore more performant.
#
# We make sure a valid directory to store virtual environments exists
# under the path /scratch/YOUR_USERNAME/virtual_environments
#
# If you call this script on your local computer (e.g, hostname != cn99, cn47 or cn48)
# the virtual environment will just be created in the root directory of this project.

VENV_DIR=/scratch/$USER/virtual_environments/statml

mkdir -p "$VENV_DIR"



# # create the virtual environment
# python3 -m venv "$VENV_DIR"

# # create a symlink to the 'venv' folder if we're on the cluster
# if [ ! -f "$PROJECT_DIR"/venv ]; then
#   ln -sfn "$VENV_DIR" "$PROJECT_DIR"/venv
# fi

# # install the dependencies
# source "$VENV_DIR"/bin/activate
# python3 -m pip install --upgrade pip
# python3 -m pip install -r "$PROJECT_DIR"/requirements.txt