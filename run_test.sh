#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -p short
#SBATCH --gpus=1

module load Python3.10 Xvfb freeglut glew
source $HOME/doc/$1/bin/activate

python test.py --gym-id $2 $3
