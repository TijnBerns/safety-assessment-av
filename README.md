# Improving Safety Assessment Of AVs By Incorporating Event Data
This repository contains the code used in the experiments described in my thesis titled TITLE.


## Downloading the data and setting up the virtual environment


## Training a flow network
During training the 50 best performing models are saved to `lightning_logs/version_VERSION`. 
At the end of the training procedure, a tensor (in `.pt` format) is saved containing the log-likelihood values of the test data.


## Computing log-likelihood and mean squared error.
To evaluate a normalizing flow, run the `./scripts/eval_tp1 VERSION` where `VERSION` is replaced with the version of the checkpoint.

Pytorch lightning automatically saves model checkpoints as `lightning_logs/version_VERSION`. When running the eval script, we look for the best checkpoint in the `lightning_logs/version` directory that is automatically created when training a model.

Author:\
BSc. Tijn Berns

Supervisor/Assessor TNO:\
Dr. Ir. Erwin de Gelder

Supervisor/Assessor Radboud:\
Dr. Yuliya Shapovalova

Second Assessor Radboud:\
Dr. Ir. Tom Claassen
