from src.train import main
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train LALM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['tensorboard', 'wandb'], help='Logger type')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging')
    parser.add_argument('--save_dir', type=str, default='./logs', help='Directory to save logs and checkpoints')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)