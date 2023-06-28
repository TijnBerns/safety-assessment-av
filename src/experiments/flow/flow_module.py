
from typing import Any
import pytorch_lightning as pl
# from nde.flows.base import Flow
import torch
from torchmetrics import MeanMetric, MeanSquaredError
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nflows import flows, distributions, transforms
from nflows.nn.nets import ResidualNet
from experiments.flow.parameters import Parameters
import utils
from tqdm import tqdm
from parameters import Parameters
from data.base import CustomDataset
import numpy as np
from typing import Any, List, Tuple


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
                 weight=None) -> None:
        super().__init__()
        self.features = features
        self.flow = create_flow(features, args)
        self.stage = stage
        self.batch_size = args.batch_size
        self.lr_stage_one = args.learning_rate_stage_1
        self.lr_stage_two = args.learning_rate_stage_2
        self.max_steps_stage_one = args.training_steps_stage_1
        self.max_steps_stage_two = args.training_steps_stage_2
        
        # For weighted training
        self.xi = dataset.xi
        self.threshold = dataset.threshold
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
                self.parameters(), lr=self.lr_stage_one)
            schedule = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_steps_stage_one,
                    eta_min=0),
                "interval": "step",
                "frequency": 1,
                "monitor": "val_log_density",
                "strict": True,
                "name": None,
            }
        elif self.stage == 2:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_stage_two)
            schedule = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_steps_stage_two,
                    eta_min=0),
                "interval": "step",
                "frequency": 1,
                "monitor": "val_log_density",
                "name": "lr",
            }
        elif self.stage == 3:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr_stage_two)
            return [optimizer]
        else:
            raise ValueError
 
        return [optimizer], [schedule]
    
    def set_stage(self, stage):
        assert stage <= 2 and stage > 0, f"The stage of the model must be either 1 or 2 but got {stage}"
        self.stage = stage
        self.configure_optimizers()
        
        
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
        
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        non_event = self.forward(batch[batch[:,self.xi] <= self.threshold])
        event = self.forward(batch[batch[:,self.xi] > self.threshold])

        # loss = - torch.mean(weighted_log_density)
        # constrained_weight = torch.nn.functional.sigmoid(self.unconstrained_weight)
        # regularization_term = self.alpha * (self.event_weight - constrained_weight) ** 2
        # weighted_log_density = torch.mean(non_event) + constrained_weight * torch.mean(event)
        # loss = - weighted_log_density + regularization_term
        constrained_weight = self.event_weight
        loss = - (torch.mean(non_event) + constrained_weight * torch.mean(event))
        
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
        
    
class FlowModuleSampled(FlowModule):
    def __init__(self, features: int, args: Parameters, dataset:CustomDataset , stage: int = 1, event_loader=None, n_event: int=100) -> None:
        super().__init__(features=features, args=args, dataset=dataset, stage=stage)
        self.alpha = 100
        self.event_loader: DataLoader = event_loader
        self.n_event = n_event       
    
    def on_test_epoch_start(self) -> None:
        # Resample event data from event model
        idx = np.random.choice(self.event_loader.dataset.data.shape[0], size=self.n_event, replace=False)
        new_event = self.event_loader.dataset.data[idx, :]
        
        # Get all non event data
        data =  self.train_dataloader().data
        non_event = data[data[:,self.xi] <= self.threshold]
        
        # Update the train dataloader
        train_loader = DataLoader(torch.cat((non_event, new_event), 0), shuffle=True, batch_size=self.batch_size)
        self.trainer.train_dataloader = train_loader
        return super().on_train_epoch_end()
    
        

    
    

        
    