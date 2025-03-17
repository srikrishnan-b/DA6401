import os
import sys
import argparse

sys.path.append(os.path.join(os.getcwd(), "src"))
import wandb
from model import train_wandb
import wandb
import json
import pprint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search using wandb sweeps"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="test_sweeps", help="Project name"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of runs to execute in the sweep",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with open("sweep_config.json", "r") as f:  # Reading sweep config file
        sweep_config = json.load(f)
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)  # Creating sweep
    wandb.agent(sweep_id, train_wandb, count=args.count)  # Running sweep
