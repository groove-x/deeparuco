from argparse import ArgumentParser
import os
import torch
from yolox.exp import get_exp
from yolox.data import COCODataset, DataPrefetcher
from yolox.utils import (
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_rank,
    get_world_size,
    init_dist,
    random_resize,
    synchronize
)
from yolox.core import launch
from yolox.utils.visualize import plot_labels_and_scores
from yolox.tools.train import Trainer

def make_parser():
    parser = ArgumentParser(description='DeepArUco++ detector trainer.')
    parser.add_argument('source_dir', help='where to find source images')
    parser.add_argument('run_name', help='directory of the resulting model')
    parser.add_argument('--model', '-m', help='base model to train', default='yolox-s')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    return parser

def main(exp, args):
    # Create output directory
    os.makedirs(args.run_name, exist_ok=True)
    
    # Create dataset config
    with open(f'{args.run_name}/dataset.yaml', 'w') as f:
        f.write(f'path: \'{args.source_dir}\'\n')
        f.write('train: \'train/images\'\n')
        f.write('val: \'valid/images\'\n')
        f.write('names:\n  0: \'marker\'')

    # Initialize trainer
    trainer = Trainer(
        exp=exp,
        args=args,
        data_dir=args.source_dir,
        output_dir=args.run_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    args = make_parser().parse_args()
    
    # Get YOLOX experiment
    exp = get_exp(None, args.model)
    
    # Launch training
    main(exp, args)