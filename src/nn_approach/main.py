import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)


import utils
from config import Config as cfg

import torch

from estimator import NN_Estimator, combined_estimation_pipline



def main():
    device, _ = utils.set_device()
    pattern = f"layers_{cfg.nn_num_hidden_layers}.neuros_{cfg.nn_num_hidden_nodes}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
    combined_estimation_pipline(NN_Estimator(), NN_Estimator(), root=cfg.path_estimates / 'nn_approach', device=device, pattern=pattern)


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    main()
    
    

"""

def train_test_pipeline(samples, eval_loader_norm, eval_loader_edge):

    device, _ = utils.set_device()
    results_dict = defaultdict(lambda: defaultdict(dict))

    # Construct dataloaders
    train_loader = DataLoader(
        samples, shuffle=True, batch_size=cfg.nn_batch_size, drop_last=True)
    val_loader = DataLoader(
        samples, shuffle=False, batch_size=cfg.nn_batch_size, drop_last=True)

    # # Fit models on data
    # baseline_kde = scipy.stats.gaussian_kde(d_norm[:,0][:M])
    num_layers = [3]
    num_hidden = [25, 50]
    for nl, nh in product(num_layers, num_hidden):
        # Initialize checkpointer
        pattern = f"layers_{nl}.neuros_{nh}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
        ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
        checkpointer = ModelCheckpoint(
            save_top_k=1,
            every_n_train_steps=500,
            monitor="val_mse",
            filename=pattern + ".best",
            save_last=True,
            auto_insert_metric_name=False,
        )

        # Fit the model
        trainer = pl.Trainer(max_steps=cfg.nn_training_steps,
                             inference_mode=False, callbacks=[checkpointer], accelerator=device)
        model = FeedForward(1, 1, nh, nl)
        trainer.fit(model, train_loader, val_loader)
        res_norm = trainer.test(model, eval_loader_norm)[0]['test_mse']
        res_edge = trainer.test(model, eval_loader_edge)[0]['test_mse']
        results_dict[nl][nh] = (res_norm, res_edge)
        print(f"\n\n============={(res_norm, res_edge)}=============\n\n")

    return results_dict
    
    
    # results_dicts = {}
    # distribution = cfg.distributions['bivariate_guassian_a']
    # single_distribution = cfg.single_distributions['bivariate_guassian_a']
    # p_edge = cfg.p_edge[1]
    # num_normal = cfg.num_normal[1]
    # num_edge = cfg.num_edge[1]

    # # Generate data from mv Gaussian
    # normal_data, edge_data, threshold = data.generate_data(
    #     distribution, p_edge, num_normal, num_edge)
    # combined_data = data.combine_data(
    #     normal_data, edge_data, threshold, p_edge)

    # # Label training data
    # _, bins = np.histogram(normal_data[:, 0], (num_normal + num_edge))
    # samples_norm, targets_norm = data.annotate_data(normal_data[:, 0], bins)
    # samples_edge, targets_edge = data.annotate_data(
    #     edge_data[:, 0], bins, targets_norm)
    # samples_combined, targets_combined = data.annotate_data(
    #     combined_data[:, 0], bins)

    # # Normal test set
    # step_size = (bins[1] - bins[0]) / 5
    # x_values_norm = np.arange(
    #     min(normal_data[:, 0]), max(normal_data[:, 0]), step_size)
    # # true_cdf_norm = scipy.stats.norm.cdf(x_values_norm, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    # true_cdf_norm = single_distribution.cdf(x_values_norm)
    # test_samples_norm = list(zip(list(x_values_norm), list(true_cdf_norm)))
    # eval_loader_norm = DataLoader(test_samples_norm)

    # # Edge test set
    # x_values_edge = np.arange(
    #     min(edge_data[:, 0]), max(edge_data[:, 0]), step_size)
    # # true_cdf_edge = scipy.stats.norm.cdf(x_values_edge, cfg.mu_X, np.sqrt(cfg.sigma_X_sq))
    # true_cdf_edge = single_distribution.cdf(x_values_edge)
    # test_samples_edge = list(zip(list(x_values_edge), list(true_cdf_edge)))
    # eval_loader_edge = DataLoader(test_samples_edge)

    # # Fit models on normal data
    # results_norm = train_test_pipeline(
    #     samples_norm, eval_loader_norm, eval_loader_edge)
    # results_dicts['normal'] = results_norm
    # with open("test.json", 'w+') as f:
    #     json.dump(results_dicts, f, indent=2)

    # # Fit model on combined data
    # results_edge = train_test_pipeline(
    #     samples_combined, eval_loader_norm, eval_loader_edge)
    # results_dicts['normal+edge'] = results_edge
    # with open("test.json", 'w+') as f:
    #     json.dump(results_dicts, f, indent=2)
    
# def construct_dataloaders(normal_data, combined_data, batch_size=100, shuffle=True):
#     # Label training data
#     _, bins = np.histogram(normal_data[:, 0], (len(normal_data) + len(combined_data)))
#     samples_norm, _ = data_utils.annotate_data(normal_data[:, 0], bins)
#     samples_combined, _ = data_utils.annotate_data(combined_data[:, 0], bins)
    
#     # construct dataloaders
#     baseline_loader = DataLoader(samples_norm, shuffle=shuffle, batch_size=cfg.nn_batch_size, drop_last=True)
#     improved_loader = DataLoader(samples_combined, shuffle=shuffle, batch_size=cfg.nn_batch_size, drop_last=True)
#     return baseline_loader, improved_loader
    
# #TODO: Solve code duplicate issue by writing wrapper class/method for estimation pipeline 
# def main():
#     # Initialize estimator
#     nl = 3
#     nh = 25
#     device, _ = utils.set_device()

#     # Define evaluation interval
#     x_values = cfg.evaluation_interval

#     for distribution_str, distribution in cfg.distributions.items():
#         true = [cfg.single_distributions[distribution_str].pdf(
#             x) for x in x_values]

#         for p_edge, num_normal, num_edge in product(cfg.p_edge, cfg.num_normal, cfg.num_edge):
#             thresholds = {}

#             # Initialize dicts to store results
#             baseline_estimates = {"x": x_values, "true": true}
#             improved_estimates = {"x": x_values, "true": true}

#             for run in tqdm(range(cfg.num_estimates), desc=f'{distribution_str}: norm={num_normal} edge={num_edge} p_edge={p_edge}'):
#                 # Initialize models
#                 nn_baseline_estimator = NN_Estimator()
#                 nn_improved_estimator = NN_Estimator()
                
#                 # Generate data
#                 normal_data, edge_data, threshold = data_utils.generate_data(distribution, p_edge, num_normal, num_edge)
                
#                 # Combine the data maintaining frac edge
#                 # p_edge_estimate = data.compute_p_edge(normal_data, threshold, dim=-1)
#                 combined_data = data_utils.combine_data(normal_data, edge_data, threshold, p_edge)
#                 baseline_loader, combined_loader = construct_dataloaders(normal_data, combined_data)
                
#                 # Fit the models
#                 pattern = f"layers_{nl}.neuros_{nh}.epoch_{{epoch:04d}}.step_{{step:09d}}.val-mse_{{val_mse:.4f}}"
#                 nn_baseline_estimator.fit(baseline_loader, baseline_loader, device, pattern)
#                 nn_improved_estimator.fit(combined_loader, combined_loader, device, pattern)         

#                 # Obtain estimates
#                 baseline_estimates[f'run_{run}'] = nn_baseline_estimator.estimate(x_values)
#                 improved_estimates[f'run_{run}'] = nn_improved_estimator.estimate(x_values)
#                 thresholds[f"run_{run}"] = threshold

#                 # Store results
#                 parent = cfg.path_estimates / 'nn_approach' /  distribution_str / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}'
#                 utils.save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.baseline.csv', df=pd.DataFrame(baseline_estimates))
#                 utils.save_csv(path=parent / f'p_edge_{p_edge}.n_normal_{num_normal}.n_edge_{num_edge}.improved.csv', df=pd.DataFrame(improved_estimates))

        
"""