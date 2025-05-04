#!/bin/bash
#BSUB -J graph_gen[1-6]          # ← ARRAY 1‑6
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -o logs/%J_%I.out          # %I = array index
#BSUB -e logs/%J_%I.err
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"

module load cuda/11.8
source ~/Desktop/AML/aml_new/bin/activate

# ---------------- parameter lookup ----------------
case $LSB_JOBINDEX in
  1) DEC="gnn" LAY=1 HEADS=0 NEG=3 BETA=0.8 ;;
  2) DEC="gnn" LAY=2 HEADS=0 NEG=3 BETA=0.8 ;;
  3) DEC="gat" LAY=1 HEADS=4 NEG=3 BETA=0.8 ;;
  4) DEC="gat" LAY=1 HEADS=8 NEG=3 BETA=0.8 ;;
  5) DEC="gat" LAY=2 HEADS=4 NEG=3 BETA=1.0 ;;
  6) DEC="gat" LAY=2 HEADS=8 NEG=3 BETA=1.0 ;;
  *) echo "Bad index $LSB_JOBINDEX"; exit 1 ;;
esac

# heads flag only for gat
EXTRA=""
if [ "$DEC" = "gat" ]; then
    EXTRA="--heads $HEADS"
fi

# ---------------- run ----------------
python -u src/main.py \
       --mode train \
       --epochs 500 \
       --lr 5e-4 \
       --hidden_dim 64 \
       --latent_dim 32 \
       --num_enc_MP_rounds 3 \
       --decoder $DEC \
       --dec_layers $LAY \
       --neg_factor $NEG \
       --beta_max $BETA \
       $EXTRA
