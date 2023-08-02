# Improving Safety Assessment Of AVs By Incorporating Event Data
This repository contains the code used in the experiments described in my thesis titled TITLE.

## Setting up the virtual environment and downloading the data 
A virtual environment can be set up by running `./scripts/setup_venv.sh`. This will create a virtual environment in the root directory named `statml` with all packages listed in `requirements.txt`. To activate the virtual environment run `source venv/bin/activate`.

Before downloading data and running experiments, ensure the `DATAROOT` variable is set to a path you want the data to be downloaded to. This can be done via ```export DATAROOT=$PATH```.

Having set the environment variable and created a virtual environment, the data can be downloaded and preprocessed by running `python src/data/preprocess.py --download=True` in the virtual/conda environment. 


## Training a flow network
To train a normalizing flow run `python src/flow/train.py --dataset=$DATASET --dataset_type=$DATASET_TYPE`. The `dataset` argument specifies which dataset to train on. Possible choices for this argument are gas, power, miniboone, and hepmass. The `dataset_type` argument specifies what type of data the normalizing flow is trained on. The following strings can be provided to this argument:
- `all`: Train on the entire training set of the UCI dataset. This option is used to obtain the normalizing flows representing the true densities.
- `normal`: Train on normal data only. This option is used as our baseline when evaluating using the log-likelihood.
- `weighted`: Train on normal and event data. This option is used in our weighted-training approach when evaluating using the log-likelihood.
- `normal_sampled`: Train on sampled normal data. This option is used as our baseline when evaluating using the MSE.
- `weighted_sampled`: rain on sampled normal and sampled event data. This option is used in our weighted-training approach when evaluating using the MSE.
  
NOTE: We can only train on sampled data once data has been sampled. 

A third parameter that can optionally be provided to `train.py` is the weight (`--weight=$WEIGHT`). This parameter specified the weight that is used during weighted training. Note that this should be a float value and it is only used when the dataset type is either set to `weighted` or `weighted_sampled`. If not provided the weight is set to 1.

All hyperparameters that are used during network training are denoted in `python/flow/parameters.py` and can be adapted accordingly.

During training the 50 best performing models are saved to `lightning_logs/version_$VERSION`. 
At the end of the training procedure, a tensor (in `.pt` format) is saved containing the log-likelihood values of the test data.

## Sampling data
Once a flow network is trained we can sample data from it by running `python src/flow/sample.py --dataset=$DATASET --version=$VERSION --num_normal=$N --num_event=$M`, where N denotes the number of normal observations that are sampled, and M the number of event observations that are sampled. If N and M are not given, the same amount of normal and event observations are drawn as in the original train sets.

When sampling the sampled data is saved to `DATAROOT` allowing us to train normalizing flows with `dataset_type` parameter set to `normal_sampled` or `weighted_sampled`.

## Computing log-likelihood and mean squared error.
To evaluate a normalizing flow, run `python src/flow/compute_llh.py --dataset=$DATASET --version=$VERSION --true=--$TRUE`, where `$DATASET` is replaced with the dataset that is used, `$VERSION` with the version of the checkpoint we wish to evaluate, and `$TRUE` with the version of the model representing the true density. The `true` parameter is only required when wanting to compute the MSE.

NOTE: Pytorch lightning automatically saves model checkpoints as `lightning_logs/version_$VERSION`. When running the eval script, we look for files ending with `llh.pt` in the `lightning_logs/version_$VERSION` directory that is automatically created when training a model. Make sure this directory exists, and tensors representing the log-likelihoods are properly saved here after network training. 

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



