
import sys
sys.path.append('src')

from typing import Any
import pytorch_lightning as pl
# from nde.flows.base import Flow
import torch
from torchmetrics import MeanMetric
from torch.utils.data import DataLoader
import torch.nn.functional as F
from nflows import flows, distributions, transforms
from nflows.nn.nets import ResidualNet
from flow.parameters import Parameters
import utils
from tqdm import tqdm
from data.base import CustomDataset
from typing import Any, List, Tuple, Dict





def create_linear_transform(features: int):
    """Creates a linear transform using LU decomposition.

    Args:
        features (int): Number of variables/features of the data.

    Returns:
        transforms.CompositeTransform: Linear tranform.
    """
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=features),
        transforms.LULinear(features, identity_init=True)
    ])

def create_base_transform(features: int, i: int, args: Parameters):
    """Creates a piecewise rational quadratic spline coupling transform.

    Args:
        features (int): Number of variables/features of the data.
        i (int): Index of the flow step.
        args (Parameters): Parameters dataclass containing hyperparamters.

    Returns:
        transforms.PiecewiseRationalQuadraticCouplingTransform: The transform
    """
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
    """Creates a transform consisting of linear tranforms and 
    piecewise rational quadratic spline coupling transforms.

    Args:
        features (int): Number of variables/features of the data
        args (Parameters): Parameters dataclass containing hyperparamters.

    Returns:
        transofrms.CompositeTransform: Transformation used in flow network.
    """
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
    """Creates a normalizing flow with a standard normal base distribution and a
    piecewise rational quadratic spline coupling transform.

    Args:
        features (int): Number of variables/features of the data.
        args (Parameters): Parameters dataclass containing hyperparamters.

    Returns:
        flows.Flow: Normalizing flow network.
    """
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
        """Initializes a LightningModule containing a flow network.

        Args:
            features (int): Number of features of the data.
            dataset (CustomDataset): Dataset that is used to train on. Not used and will be deprecated.
            args (Parameters): _description_
            stage (int, optional): _description_. Defaults to 1. The training stage. If 1 use a cosine annealing 
            weight (float, optional): _description_. Defaults to 1.0. Not used and will be deprecated.
        """
        super().__init__()
        # Features and hyperparamters
        self.features = features
        self.stage = stage
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.max_steps = args.training_steps
        
        # Construct flow network
        self.flow = create_flow(features, args)
                
        # Metrics
        self.train_mean_log_density: MeanMetric = MeanMetric()
        self.val_mean_log_density: MeanMetric = MeanMetric()
        
    def forward(self, batch, **kwargs):
        batch = batch.float()
        return self.flow.log_prob(batch)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        log_density = self.forward(batch)
        loss = - torch.mean(log_density)
        self.train_mean_log_density(log_density)
        self.log("log_density", -loss, batch_size=self.batch_size, prog_bar=True)
        return loss
        
    def on_train_epoch_end(self) -> None:
        self.log("train_mean_log_density", self.train_mean_log_density, prog_bar=False)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        log_density = self.forward(batch)
        loss = - torch.mean(log_density)
        self.val_mean_log_density(log_density)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val_log_density", self.val_mean_log_density, prog_bar=True)
        
    def compute_llh(self, dataloader: DataLoader):
        """Returns a tensor of the LLH of each of the observations in the provided dataloader.

        Args:
            dataloader (DataLoader): Dataloader containing observations for which the llh is computed.

        Returns:
            torch.Tensor: Tensor containing LLh values
        """
        llh = torch.zeros(len(dataloader.dataset.data))
        with torch.no_grad():
            i = 0
            for batch in tqdm(dataloader):
                batch  = batch.to(self.device)
                llh[i:i + len(batch)] = self.forward(batch).to('cpu')
                i += len(batch)
        return llh
                
                        
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the flow network.

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Tensor of samples drawn from flow network.
        """
        with torch.no_grad():
            return self.flow.sample(num_samples)
        
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
        self.xi = dataset.xi
        self.threshold = dataset._threshold
        self.weight = weight
        
    def _compute_weighted_loss(self, non_event, event):
        return - torch.mean(torch.cat((non_event, self.weight * event)))
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        non_event = self.forward(batch[batch[:,self.xi] <= self.threshold])
        event = self.forward(batch[batch[:,self.xi] > self.threshold])
        
        loss = self._compute_weighted_loss(non_event, event)
        log_density = torch.mean(torch.cat((non_event, event)))
        
        self.train_mean_log_density(log_density)
        self.log("weighted_log_density", -loss, batch_size=self.batch_size)
        
        log_density = torch.mean(torch.cat((non_event, event)))
        self.log("log_density", log_density, batch_size=self.batch_size, prog_bar=True)
        self.log('event', torch.mean(event))
        self.log('non_event', torch.mean(non_event))
        self.log('event_weight', self.weight)
        return loss
    
        
        
class FlowModuleTrainableWeight(FlowModuleWeighted):
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes FlowModuleWeighted using a dictionary of parameters (used in population based training).

        Args:
            config (Dict[str, Any]): Parameters used to initialize FlowModuleWeighted.
        """
        super().__init__(**config)    
    