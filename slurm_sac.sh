#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH --gpus=1
#SBATCH -p short
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
ENV_NAME = $1
module load Python3.10 Xvfb freeglut glew
source $HOME/doc/Planning-the-path-with-rl/$ENV_NAME/bin/activate

# Run the script
python train_sac.py --cuda --gym-id $2 --track --capture-video
