import sys
sys.path.append('src')

import data.data_utils as data_utils
import utils
from pathlib import Path

import torch
import click
from torch.utils.data import DataLoader
from flow_module import FlowModule
from config import MVParameters as mv_params
from config import FlowParameters as flow_params
import matplotlib.pyplot as plt

from data.gas import Gas
from data.power import Power

@click.command()
@click.option('--checkpoint', type=str)
def main(checkpoint: str):
    normal  = Power(split='test_normal')
    event = Power(split='test_event')
    test = Power(split='test')
    features = normal.data.shape[1]
    
    normal = DataLoader(normal, batch_size=flow_params.batch_size)
    event = DataLoader(event, batch_size=flow_params.batch_size)
    test = DataLoader(test, batch_size=flow_params.batch_size)
    
    
    # Load model from checkpoint
    flow_module = FlowModule.load_from_checkpoint(checkpoint, features=features).eval()

    # # Generate/load data
    # distributions_, _, distribution = mv_params.get_distributions()
    # threshold = data_utils.determine_threshold(flow_params.p_event, distributions_[-1])
    # normal_data, event_data = data_utils.generate_data(distribution, flow_params.num_normal, flow_params.num_event, threshold, random_state=2023)
    # normal_data_only, event_data_only = data_utils.filter_data(normal_data, event_data)
    
    
    # # Convert to tensors
    # normal_data = torch.Tensor(normal_data)
    # event_data = torch.Tensor(event_data)
    # normal_data_only = torch.Tensor(normal_data_only)
    # event_data_only = torch.Tensor(event_data_only)
    
    # Print log likelihood for normal and event data
    print(f'log likelihood all {flow_module.compute_log_prob(test)}')
    print(f'log likelihood normal {flow_module.compute_log_prob(normal)}')
    print(f'log likelihood event {flow_module.compute_log_prob(event)}')

    # # Create contour plots of true and estimated pdf
    # x_values = torch.tensor(data_utils.get_evaluation_interval(distributions_, n=200), dtype=torch.float)
    # xline = torch.unique(x_values[:,0])
    # yline = torch.unique(x_values[:,1])
    # xgrid, ygrid = torch.meshgrid(xline, yline)
    
    # true = torch.Tensor(distribution.pdf(x_values).reshape(len(xline),len(yline)))
    # pdf_estimate = flow_module.compute_pdf(x_values).reshape(len(xline),len(yline))
    # print(f'MSE {((true - pdf_estimate) ** 2).mean()}')
    
    # _, ax = plt.subplots(1, 2)
    # ax[0].contourf(xgrid.numpy(), ygrid.numpy(), true.numpy())
    # ax[1].contourf(xgrid.numpy(), ygrid.numpy(), pdf_estimate.numpy())
    # # plt.show()
    
    

if __name__ == "__main__":
    main()