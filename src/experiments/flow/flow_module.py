from typing import Any
import pytorch_lightning as pl
# from nde.flows.base import Flow
import torch
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nflows import flows, distributions, transforms
from nflows.nn.nets import ResidualNet
import utils
from config import FlowParameters as args
from torch import nn

def create_linear_transform(features):
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=features),
        transforms.LULinear(features, identity_init=True)
    ])

def create_base_transform(features, i):
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(features, even=(i % 2 == 0)),
        transform_net_create_fn=lambda in_features, out_features: ResidualNet(
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

def create_transform(features):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(features),
            create_base_transform(features, i)
        ]) for i in range(args.num_flow_steps)
    ] + [
        create_linear_transform(features)
    ])
    return transform

# def create_flow(features):
#     # create model
#     flow_distribution = distributions.StandardNormal((features,))
#     transform = create_transform(features)
#     return flows.Flow(transform, flow_distribution)

def create_flow(features):
    num_layers = 5
    base_dist = distributions.StandardNormal((features,))

    transform = []
    for _ in range(num_layers):
        transform.append(transforms.ReversePermutation(features=2))
        transform.append(transforms.MaskedAffineAutoregressiveTransform(features=features, 
                                                            hidden_features=4, 
                                                            context_features=0))
    transform = transforms.CompositeTransform(transform)

    flow = flows.Flow(transform, base_dist)
    return flow

class FlowModule(pl.LightningModule):
    def __init__(self,  
                 lr_stage_one: float = args.learning_rate_stage_1, 
                 lr_stage_two: float = args.learning_rate_stage_2, 
                 max_steps_stage_one: int = args.training_steps_stage_1, 
                 max_steps_stage_two: int = args.training_steps_stage_2, 
                 features: int = args.features,
                 batch_size: int =args.batch_size,
                 stage: int=1) -> None:
        super().__init__()
        self.features = features
        self.flow = create_flow(features)
        self.stage = stage
        self.batch_size = batch_size
        self.lr_stage_one = lr_stage_one
        self.lr_stage_two = lr_stage_two
        self.max_steps_stage_one = max_steps_stage_one
        self.max_steps_stage_two = max_steps_stage_two
        
        # Metrics
        self.train_mean_log_density: MeanMetric = MeanMetric()
        self.val_mean_log_density: MeanMetric = MeanMetric()
        self.test_mean_log_density: MeanMetric = MeanMetric()
        
    def forward(self, batch, **kwargs):
        batch = batch.float()
        return self.flow.log_prob(batch)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        log_density = self.forward(batch)
        loss = - torch.mean(log_density)
        self.train_mean_log_density(-loss)
        self.log("log_density", -loss, batch_size=self.batch_size, prog_bar=True)
        return loss
        
    def on_train_epoch_end(self) -> None:
        self.log("train_mean_log_density", self.train_mean_log_density, prog_bar=False)
        return

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        log_density = self.forward(batch)
        loss = - torch.mean(log_density)
        self.val_mean_log_density(-loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_log_density", self.val_mean_log_density, prog_bar=True)
        return

    def test_step(self, batch, batch_idx) -> None:
        log_density = self.forward(batch)
        loss = - torch.mean(log_density)
        self.test_mean_log_density(-loss)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test_log_density", self.test_mean_log_density)
        return
    
    def compute_pdf(self, x_values: torch.Tensor):
        with torch.no_grad():
            prob = self.flow.log_prob(x_values).exp()
                
        return prob
    
    def compute_log_prob(self, x_values: torch.Tensor):
        x_values =x_values.float()
        with torch.no_grad():
            log_prob = self.flow.log_prob(x_values)
            
        return log_prob.mean(), log_prob.std()


    def configure_optimizers(self) -> None:
        # setup the optimization algorithm
        if self.stage == 1:
            optimizer = torch.optim.Adam(
                self.flow.parameters(), lr=self.lr_stage_one)
            # schedule = {
            #     "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer,
            #         T_max=self.max_steps_stage_one,
            #         eta_min=0),
            #     "interval": "step",
            #     "frequency": 1,
            #     "monitor": "val_log_density",
            #     "strict": True,
            #     "name": None,
            # }
        elif self.stage == 2:
            optimizer = torch.optim.Adam(
                self.flow.parameters(), lr=self.lr_stage_two)
            # setup the learning rate schedule.
            # schedule = {
            #     # Required: the scheduler instance.edu
            #     "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            #         optimizer,
            #         T_max=self.max_steps_stage_two,
            #         eta_min=0),
            #     # The unit of the scheduler's step size, could also be 'step'.
            #     # 'epoch' updates the scheduler on epoch end whereas 'step'
            #     # updates it after an optimizer update.    # def _extract_embeddings_batch(self, hidden_states: torch.Tensor, processed_logits: torch.Tensor):
            #     "interval": "step",
            #     # How many epochs/steps should pass between calls to
            #     # `scheduler.step()`. 1 corresponds to updating the learning
            #     # rate after every epoch/step.
            #     "frequency": 1,
            #     "monitor": "val_log_density",
            #     # If using the `LearningRateMonitor` callback to monitor the
            #     # learning rate progress, this keyword can be used to specify
            #     # a custom logged nameself.stage = stage
            #     "name": "lr",
            # }
        else:
            raise ValueError

        # return [optimizer], [schedule]
        return [optimizer]
    
    def set_stage(self, stage):
        assert stage <= 2 and stage > 0, f"The stage of the model must be either 1 or 2 but got {stage}"
        self.stage = stage
        self.configure_optimizers()

        
    