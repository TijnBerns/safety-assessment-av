import sys
sys.path.append('src')

import data.data_utils as data_utils
import utils
from pathlib import Path


import click
from torch.utils.data import DataLoader
from flow_module import FlowModule
from config import MVParameters as mv_params
from config import FlowParameters as flow_params
import matplotlib.pyplot as plt
import parameters
from utils import save_json
import pandas as pd


def write_results(row):
    results_file = Path('results.csv')
    if not results_file.exists():
        results_file.touch()
        with open(results_file, 'w') as f:
            f.write('version, all, normal, event')
    df = pd.read_csv(results_file, index_col=False)
    df.loc[-1] = row
    df.to_csv(results_file, index=False)
    
def sort_fn(path: Path):
    s = path.name.split('.')
    r = s[-3]
    m = s[-4][-3:]
    if m[0] == '_':
        m = m[1:]
    return float(f'{m}.{r}')
    

def get_checkpoint(version):
    path = Path(f'/home/tberns/safety-assessment-av/lightning_logs/version_{version}/checkpoints')
    best_checkpoint = list(path.rglob('*best.ckpt'))
    best_checkpoint.sort(key=sort_fn)

    # last_checkpoint = list(path.rglob('*last.ckpt'))[0]
    return best_checkpoint[-1], None


def eval(checkpoint, version, dataset):
    # Set device
    device, _ = utils.set_device()
    
    # Retrieve arguments correspondnig to dataset
    args = parameters.get_parameters(dataset)
    
    # Inititialize datasets
    dataset = parameters.get_dataset(dataset)
    normal  = dataset(split='test_normal')
    event = dataset(split='test_event')
    test = dataset(split='test')
    _test = dataset(split='_test')
    features = normal.data.shape[1]
    
    # Initialize dataloaders
    normal = DataLoader(normal, batch_size=flow_params.batch_size)
    event = DataLoader(event, batch_size=flow_params.batch_size)
    test = DataLoader(test, batch_size=flow_params.batch_size)
    _test = DataLoader(_test, batch_size=flow_params.batch_size)
    
    # Load model from checkpoint
    flow_module = FlowModule.load_from_checkpoint(checkpoint, features=features, device=device, args=args, dataset=dataset()).eval()
    flow_module = flow_module.to(device)

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
    llh_all = float(flow_module.compute_log_prob(test))
    print(f'log likelihood all {llh_all}')
    
    llh_normal = float(flow_module.compute_log_prob(normal))
    print(f'log likelihood normal {llh_normal}')
    
    llh_event = float(flow_module.compute_log_prob(event))
    print(f'log likelihood event {llh_event}')   
    
    # Store results
    write_results([version, llh_all, llh_normal, llh_event])
    
    # print(f'log likelihood normal {flow_module.compute_log_prob(normal)}')
    # print(f'log likelihood all (normalized using all train data) {flow_module.compute_log_prob(_test)}')

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
    return version, llh_all, llh_normal, llh_event
    


@click.command()
@click.option('--version', type=str)
@click.option('--dataset', default='hepmass')
def main(version: str, dataset: str):
    best, last = get_checkpoint(version)
    
    # evaluate best checkpoint
    print(f'Evaluating {version} best')
    eval(best,version + ' best', dataset)
    
    # # Evaluate last checkpoint
    # print(f'Evaluating {version} last')
    # eval(last, version + ' last',dataset)
    

if __name__ == "__main__":
    main()