#!/bin/bash
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=mgm4@cin.ufpe.br

ENV_NAME=$1
module load Python3.10
python -m venv $HOME/doc/$ENV_NAME
source $HOME/doc/$ENV_NAME/bin/activate
which python
pip install -r requirements.txt
pip list
