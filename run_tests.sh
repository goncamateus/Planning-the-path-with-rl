#!/bin/bash

sbatch run_test.sh c1 Baseline-v0
sbatch run_test.sh c1 Baseline-v0 --caps
sbatch run_test.sh c1 Baseline-v1
sbatch run_test.sh c1 Baseline-v1 --caps
sbatch run_test.sh c1 Enhanced-v0
sbatch run_test.sh c1 Enhanced-v0 --caps
sbatch run_test.sh c1 Enhanced-v1
sbatch run_test.sh c1 Enhanced-v1 --caps
sbatch run_test.sh c1 Obstacle-v1
sbatch run_test.sh c1 Obstacle-v1 --caps
