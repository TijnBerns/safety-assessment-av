from argparse import ArgumentParser
from config import MVParameters as parameters
import data.data_utils as data_utils
from torch.utils.data import DataLoader

import scipy
import scipy.stats



from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
import data.data_utils as data_utils

from nde import distributions, flows, transforms

import nn as nn_
import utils
import torch.functional as F
parser = ArgumentParser()


parser.add_argument('--num_normal', default=int(10e3))
parser.add_argument('--num_event', default=int(10e3))
parser.add_argument('--p_event', default=0.08)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--training_steps_stage_1', default=int(10e3))
parser.add_argument('--training_steps_stage_2', default=int(10e3))
parser.add_argument('--learning_rate_stage_1', default=1e-4)
parser.add_argument('--learning_rate_stage_2', default=1e-4)
parser.add_argument('--logging_interval', default=50)
# flow details
parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
                    choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
                             'affine-autoregressive', 'quadratic-autoregressive',
                             'rq-autoregressive'],
                    help='Type of transform to use between linear layers.')
parser.add_argument('--linear_transform_type', type=str, default='lu',
                    choices=['permutation', 'lu', 'svd'],
                    help='Type of linear transform to use.')
parser.add_argument('--num_flow_steps', type=int, default=10,
                    help='Number of blocks to use in flow.')
parser.add_argument('--hidden_features', type=int, default=256,
                    help='Number of hidden features to use in coupling/autoregressive nets.')
parser.add_argument('--tail_bound', type=float, default=3,
                    help='Box is on [-bound, bound]^2')
parser.add_argument('--num_bins', type=int, default=8,
                    help='Number of bins to use for piecewise transforms.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks to use in coupling/autoregressive nets.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in coupling/autoregressive nets.')
parser.add_argument('--dropout_probability', type=float, default=0.25,
                    help='Dropout probability for coupling/autoregressive nets.')
parser.add_argument('--apply_unconditional_transform', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to unconditionally transform \'identity\' '
                         'features in coupling layer.')

args = parser.parse_args()


def create_linear_transform(features):
    if args.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif args.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif args.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


def create_base_transform(features, i):
    if args.base_transform_type == 'affine-coupling':
        return transforms.AffineCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            )
        )
    elif args.base_transform_type == 'quadratic-coupling':
        return transforms.PiecewiseQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'rq-coupling':
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.hidden_features,
                context_features=None,
                num_blocks=args.num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
    elif args.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'quadratic-autoregressive':
        return transforms.MaskedPiecewiseQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    elif args.base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm
        )
    else:
        raise ValueError


def create_transform(features):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(features),
            create_base_transform(features)
        ]) for i in range(args.num_flow_steps)
    ] + [
        create_linear_transform()
    ])
    return transform

def main():
    # Generate/load data
    distributions_, _, distribution = parameters.get_distributions()
    threshold = data_utils.determine_threshold(args.p_event, distributions[-1])
    normal_data, event_data = data_utils.generate_data(distribution, args.num_normal, args.num_event, threshold)
    features = normal_data.dim
    
    # Construct data loaders
    normal_loader = DataLoader(normal_data, shuffle=True, batch_size=args.batch_size)
    event_loader = DataLoader(normal_data, shuffle=True, batch_size=args.batch_size)
    
    # create model
    distribution = distributions.StandardNormal((features,))
    transform = create_transform(features)
    flow = flows.Flow(transform, distribution).to(device)
    model = flow # TODO
    
    # Define model and device
    device, jobid = utils.set_device()    
    
    # Initialize checkpointer
    pattern = ''
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        every_n_train_steps=100,
        monitor="val_mse",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Pre-train on event data
    trainer_stage_1 = pl.Trainer(max_steps=args.training_steps_stage_1,
                            inference_mode=False,
                            callbacks=[checkpointer],
                            #  logger=False,
                            log_every_n_steps=args.logging_interval,
                            accelerator=device)
    trainer_stage_1.fit(model, event_loader)
    
    
    # Fine-tune on normal data
    trainer_stage_2 = pl.Trainer(max_steps=args.training_steps_stage_2,
                        inference_mode=False,
                        callbacks=[checkpointer],
                        # logger=False,
                        log_every_n_steps=args.logging_interval,
                        accelerator=device)
    trainer_stage_2.fit(model, normal_loader)

if __name__ == "__main__":
    main()