
import sys
sys.path.append('src')

from typing import Any
import pytorch_lightning as pl
# from nde.flows.base import Flow
import torch
from torchmetrics import MeanMetric, MeanSquaredError
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nflows import flows, distributions, transforms
from nflows.nn.nets import ResidualNet
from flow.parameters import Parameters
import utils
from tqdm import tqdm
from data.base import CustomDataset
import numpy as np
from typing import Any, List, Tuple
from data.power import Power
from data.gas import Gas
import scipy.special



def create_linear_transform(features):
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=features),
        transforms.LULinear(features, identity_init=True)
    ])

def create_base_transform(features: int, i: int, args: Parameters):
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

def create_transform(features: int, args: Parameters):
    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(features),
            create_base_transform(features, i, args)
        ]) for i in range(args.num_flow_steps)
    ] + [
        create_linear_transform(features)
    ])
    return transform

def create_flow(features: int, args: Parameters):
    # create model
    flow_distribution = distributions.StandardNormal((features,))
    transform = create_transform(features, args)
    return flows.Flow(transform, flow_distribution)

class FlowModule(pl.LightningModule):
    def __init__(self,  
                 features: int,
                 dataset: CustomDataset,
                 args: Parameters, 
                 stage: int=1,
                 weight: float = 1.0) -> None:
        super().__init__()
        self.features = features
        self.flow = create_flow(features, args)
        self.stage = stage
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.max_steps = args.training_steps
        
        # For weighted training
        self.xi = dataset.xi
        self.threshold = dataset._threshold
        # self.event_weight = dataset.weight
        self.event_weight = weight
                
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
    
    def compute_llh(self, dataloader: DataLoader):
        llh = torch.zeros(len(dataloader.dataset.data))
        with torch.no_grad():
            i = 0
            for batch in tqdm(dataloader):
                batch  = batch.to(self.device)
                llh[i:i + len(batch)] = self.forward(batch).to('cpu')
                i += len(batch)
        return llh
                
    
    def compute_pdf(self, x_values: torch.Tensor):
        with torch.no_grad():
            prob = self.flow.log_prob(x_values).exp()
                
        return prob
    
    def compute_log_prob(self, dataloader: DataLoader):
        log_prob = MeanMetric().to(self.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                log_prob.update(self.forward(batch))
            
        return log_prob.compute()
    
    def compute_mse(self, other: 'FlowModule', dataloader: DataLoader):
        """Computes the mean squared error (MSE) of this module and another flow module.

        Args:
            other (FlowModule):The flow module used as a comparison.
            dataloader (DataLoader): Dataloader used for computations.

        Returns:
            float: MSE between this flow module and other.
        """
        mse = MeanSquaredError().to(self.device)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(self.device)
                target = self.forward(batch)
                inputs = other.forward(batch)
                mse.update(inputs, target)
        return mse.compute()
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the flow network.

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Tensor of samples drawn from flow network.
        """
        with torch.no_grad():
            return self.flow.sample(num_samples)
        
    def freeze_partially(self):
        """Freezes half of the flow layers.
        """
        named_modules = list(self.flow._transform._transforms.named_children())
        for i in range(len(named_modules) // 2):
            named_modules[i][1].requires_grad_(False)
        
    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Initializes the optimizer and learning rate scheduler.

        Raises:
            ValueError: Raises error if the stage of the module is not set to 1 or 2.

        Returns:
            _type_: _description_
        """
        # setup the optimization algorithm
        if self.stage == 1:
            optimizer = torch.optim.Adam(
                self.flow.parameters(), lr=self.lr)
            schedule = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_steps,
                    eta_min=0),
                "interval": "step",
                "frequency": 1,
                "monitor": "val_log_density",
                "strict": True,
                "name": None,
            }
            return [optimizer], [schedule]
        elif self.stage == 2:
            optimizer = torch.optim.Adam(
                self.flow.parameters(), lr=self.lr)
            return [optimizer]
        else:
            raise ValueError        
        
class FlowModuleWeighted(FlowModule):
    def __init__(self,  
                 features: int,
                 dataset: CustomDataset,
                 args: Parameters, 
                 stage: int=1,
                 weight=None) -> None:
        super().__init__(features=features, args=args, dataset=dataset, stage=stage, weight=weight)
        self.alpha = 100
        # self.unconstrained_weight = torch.nn.Parameter(torch.logit(torch.tensor(self.event_weight)))
        # self.weight_optimizer = torch.optim.Adam([self.unconstrained_weight], lr=0.05)
        # self.weight_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.weight_optimizer, T_max=self.max_steps_stage_two // self.batch_size, eta_min=0)
        
    def _compute_weighted_loss(self, non_event, event):
        # constrained_weight = torch.nn.functional.sigmoid(self.event_weight)
        constrained_weight = self.event_weight
        return - torch.mean(torch.cat((non_event, constrained_weight * event)))
    
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        non_event = self.forward(batch[batch[:,self.xi] <= self.threshold])
        event = self.forward(batch[batch[:,self.xi] > self.threshold])

        # loss = - torch.mean(weighted_log_density)
        # constrained_weight = torch.nn.functional.sigmoid(self.unconstrained_weight)
        # regularization_term = self.alpha * (self.event_weight - constrained_weight) ** 2
        # weighted_log_density = torch.mean(non_event) + constrained_weight * torch.mean(event)
        # loss = - weighted_log_density + regularization_term
        constrained_weight = self.event_weight
        loss = self._compute_weighted_loss(non_event, event)
        
        self.train_mean_log_density(-loss)
        self.log("weighted_log_density", -loss, batch_size=self.batch_size)
        self.log('weight', constrained_weight)
        
        log_density = torch.mean(torch.cat((non_event, event)))
        self.log("log_density", log_density, batch_size=self.batch_size, prog_bar=True)
        self.log('event', torch.mean(event))
        self.log('non_event', torch.mean(non_event))
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        
class FlowModuleTrainableWeight(FlowModule):
    def __init__(self, config) -> None:
        # super().__init__(features=features, args=args, dataset=dataset, stage=stage, weight=weight)
        super().__init__(features=config["features"], args=config["args"], dataset=config["dataset"], stage=config["stage"], weight=config["weight"])
        
        # self.event_weight = torchtorch.logit(torch.tensor(self.event_weight))
        self.normal_dataloader = DataLoader(Gas(split='normal_sampled'), batch_size=self.batch_size, shuffle=False)
        self.event_weight = torch.nn.Parameter(torch.tensor(self.event_weight))
        # self.weight_optimizer = torch.optim.Adam([self.event_weight], lr=0.005)
    
    def _compute_weighted_loss(self, non_event, event, weight):
        # constrained_weight = torch.nn.functional.sigmoid(weight)
        return - torch.mean(torch.cat((non_event, weight * event))), weight
    
    def _compute_weighted_loss_b(self, non_event, event, weight):
        # constrained_weight = torch.nn.functional.sigmoid(weight)
        return - torch.mean(torch.cat(((1-weight) * non_event, weight * event))), weight
    
    def _compute_inverse_weighted_loss(self, non_event, event):
        constrained_weight = self.event_weight
        return - torch.mean(torch.cat((constrained_weight) * non_event, (1-constrained_weight) * event)), constrained_weight
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        # self.event_weight.requires_grad_(False)
        non_event = self.forward(batch[batch[:,self.xi] <= self.threshold])
        event = self.forward(batch[batch[:,self.xi] > self.threshold])
        
        loss, _ = self._compute_weighted_loss(non_event, event, self.event_weight)
        log_density = torch.mean(torch.cat((non_event, event)))
        
        self.train_mean_log_density(log_density)
        self.log("weighted_log_density", -loss, batch_size=self.batch_size)
        
        log_density = torch.mean(torch.cat((non_event, event)))
        self.log("log_density", log_density, batch_size=self.batch_size, prog_bar=True)
        self.log('event', torch.mean(event))
        self.log('non_event', torch.mean(non_event))
        self.log('event_weight', self.event_weight)
        return loss
    
    # def _compute_llh(self, dataloader):
    #     i = 0
    #     for batch in dataloader:
    #         batch  = batch.to(self.device)
    #         llh[i:i + len(batch)] = self.forward(batch).to('cpu')
    #         i += len(batch)
    #     return torch.mean(llh)
    
    # def on_train_epoch_end(self) -> None:
    #     log_prob_metric = MeanMetric().to(self.device)
    #     with torch.no_grad():
    #         for batch in self.normal_dataloader:
    #             batch = batch.to(self.device)
    #             log_prob = self.forward(batch)
    #             log_prob_metric(log_prob)
    #     self.log("log_prob": log_prob_metric)
    #     session.report({"log_prob": log_prob_metric.value()})
        
    
    # def on_train_epoch_start(self) -> None:
    #     # Optimize weight
    #     self.freeze()
    #     epsilon = 0.05
        
    #     temp_weight_unconstrained = scipy.special.logit(0.5)
    #     temp_weight_unconstrained = torch.tensor(temp_weight_unconstrained, device=self.device, requires_grad=True)
        
    #     for batch in self.normal_dataloader:
    #         batch = batch.to(self.device)
    #         non_event = self.forward(batch[batch[:,self.xi] <= self.threshold])
    #         event = self.forward(batch[batch[:,self.xi] > self.threshold])
            
    #         temp_weight_constrained = torch.sigmoid(temp_weight_unconstrained)
    #         loss, _ = self._compute_weighted_loss_b(non_event, event, temp_weight_constrained)
    #         loss.backward()
  
    #         temp_weight_unconstrained.data = temp_weight_unconstrained.data + epsilon * temp_weight_unconstrained.grad.detach().sign()

    #         temp_weight_unconstrained.grad.zero_()
    #         self.log("weight", temp_weight_constrained.data, on_step=True)
            
    #     self.event_weight = temp_weight_constrained.data
    #     self.unfreeze()
            

    
        
    