#!/bin/bash
#BSUB -J graph_generations
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
# end of BSUB options

module load cuda/11.8

source ~/Desktop/AML/aml_new/bin/activate

python -u src/main.py --mode 'train' --epochs 500 --lr 5e-4 --hidden_dim 64 --latent_dim 32 --num_enc_MP_rounds 3 --decoder gat  --neg_factor 3 --dec_layers 1 --heads 4 


