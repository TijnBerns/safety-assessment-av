import sys
sys.path.append('src')

import utils
import data.data_utils as data_utils
from flow_module import FlowModule
from config import FlowParameters as args
from config import MVParameters as parameters

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


def create_checkpointer():
    pattern = "epoch_{epoch:04d}.step_{step:09d}.log_density_{log_density:.2f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        every_n_train_steps=50,
        monitor="val_log_density",
        mode='max',
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )
    return checkpointer

def train_val_split(data, frac):
    split_idx = int(frac * len(data))
    return  data[:split_idx], data[split_idx:]

def create_data_loaders():
    # Generate/load data
    distributions_, _, distribution = parameters.get_distributions()
    threshold = data_utils.determine_threshold(args.p_event, distributions_[-1])
    normal_data, event_data = data_utils.generate_data(distribution, args.num_normal, args.num_event, threshold)
    _, event_data = data_utils.filter_data(normal_data, event_data, threshold)

    normal_data_train, normal_data_val = train_val_split(normal_data, args.val_frac) 
    event_data_train, event_data_val = train_val_split(event_data, args.val_frac) 
    
    return (DataLoader(normal_data_train, shuffle=True, batch_size=args.batch_size),
            DataLoader(normal_data_val, shuffle=False, batch_size=args.batch_size),
            DataLoader(event_data_train, shuffle=True,batch_size=args.batch_size),
            DataLoader(event_data_val, shuffle=False,batch_size=args.batch_size),
            ) 

@click.command()
@click.option('--pre_train', type=bool, default=True)
def train(pre_train: bool):
    # Construct data loaders
    normal_train, normal_val, event_train, event_val = create_data_loaders()

    # Get device
    device, _ = utils.set_device()

    # create model
    flow_module = FlowModule()

    # Initialize checkpointers
    checkpointer_stage_1 = create_checkpointer()
    checkpointer_stage_2 = create_checkpointer()

    if pre_train:
        # Pre-train on event data
        trainer_stage_1 = pl.Trainer(max_steps=args.training_steps_stage_1,
                                     inference_mode=False,
                                     callbacks=[checkpointer_stage_1],
                                     log_every_n_steps=args.logging_interval,
                                     accelerator=device)
        trainer_stage_1.fit(flow_module, event_train, event_val)

    # Fine-tune on normal data
    trainer_stage_2 = pl.Trainer(max_steps=args.training_steps_stage_2,
                                 inference_mode=False,
                                 callbacks=[checkpointer_stage_2],
                                 log_every_n_steps=args.logging_interval,
                                 accelerator=device)
    flow_module.set_stage(2)
    trainer_stage_2.fit(flow_module, normal_train, normal_val)


if __name__ == "__main__":
    train()
