#!/usr/bin/env python
"""
Run script for the Graph VAE project.
This script ensures the proper Python paths are set up and handles command-line arguments.

Usage:
    python run.py train   # Train the model
    python run.py eval    # Evaluate the model
"""
import os
import sys
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Parse the first argument to determine mode
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'eval']:
        print("Usage: python run.py [train|eval] [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # Remove the first argument (mode) so the subcommand can parse rest of args
    sys.argv.pop(1)
    
    if mode == 'train':
        from src.main import main
        main()
    elif mode == 'eval':
        parser = argparse.ArgumentParser(description="Evaluate Graph VAE model")
        parser.add_argument('--model_path', default='graph_vae_model.pt',
                            help='Path to model checkpoint file')
        parser.add_argument('--output_prefix', default='results',
                            help='Prefix for output files')
        
        args = parser.parse_args()
        
        from src.evaluate import evaluate
        evaluate(model_path=args.model_path, output_prefix=args.output_prefix)