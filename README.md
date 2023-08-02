# Improving Safety Assessment Of AVs By Incorporating Event Data
This repository contains the code used in the experiments described in my thesis titled TITLE.


## Setting up the virtual environment and ownloading the data 
A virtual environment can be setup by running `./scripts/setup_venv.sh`. This will create a virtual environment with all packages listed in `requirements.txt`. 

Before downloading data and running experiments, ensure the `DATAROOT` variable is set to a path you want the the data downloaded to:\
```export DATAROOT=$path```

Having set the environment variable, the data can be downloaded and preprocessed by running `python src/data/preprocess.py --download=True` in the virtual/conda environment. 


## Training a flow network
During training the 50 best performing models are saved to `lightning_logs/version_VERSION`. 
At the end of the training procedure, a tensor (in `.pt` format) is saved containing the log-likelihood values of the test data.


## Computing log-likelihood and mean squared error.
To evaluate a normalizing flow, run the `./scripts/eval_tp1 VERSION` where `VERSION` is replaced with the version of the checkpoint.

Pytorch lightning automatically saves model checkpoints as `lightning_logs/version_VERSION`. When running the eval script, we look for the best checkpoint in the `lightning_logs/version` directory that is automatically created when training a model.


## Abstract
Automated Vehicles (AVs) are expected to make a significant contribution to lowering the number of traffic accidents. With AVs responsibility of the driver is transferred to the vehicle, making assessment in terms of risk essential to the deployment of AVs. Scenario-based assessment approaches aim to provide the soundest evidence that the deployment of an AV is safe. With scenario-based assessment, the responses of an AV are tested under various given scenarios. For a complete assessment, it is important that the set of scenarios is extensive. A multitude of data-driven approaches has been proposed to provide a more complete set of scenarios with the aim to improve the assessment of AVs. The main stumbling block of this is that the models used in data-driven approaches have a tendency to sample unrealistic event scenarios such as (near) traffic accidents due to the lack of such events in the training data. This motivates the use of an additional dataset that only contains event scenarios. In this research, we explore three approaches that aim to improve the density estimates of scenario statistics by incorporating the extra set of event data in the estimate. The improved density estimates can be used to sample more realistic scenarios from, allowing for better assessment of AVs. The first approach we consider adds the event data to the training data while maintaining the correct fraction of events by duplicating scenarios from the initial training set. The second approach computes a weighted sum over an estimate made on non-event data and an estimate made on event data. The third approach adapts the loss function of normalizing flow networks by adding a weight parameter. This parameter reduces the contribution of event data to the loss, allowing us to add the event data to the training data without overestimating the density of event data. Through conducting extensive experiments in various data settings, we demonstrate that two out of the three considered approaches can successfully be applied to improve the density estimates of scenario statistics.


Author:\
BSc. Tijn Berns

Supervisor/Assessor TNO:\
Dr. Ir. Erwin de Gelder

Supervisor/Assessor Radboud:\
Dr. Yuliya Shapovalova

Second Assessor Radboud:\
Dr. Ir. Tom Claassen
