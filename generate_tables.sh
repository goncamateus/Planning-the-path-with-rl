#!/bin/bash

sbatch -w cluster-node1 -o test_0.log run_test.sh c1 Baseline-v0
sbatch -w cluster-node1 -o test_1.log run_test.sh c1 Baseline-v0 --caps
sbatch -w cluster-node2 -o test_2.log run_test.sh c1 Baseline-v1
sbatch -w cluster-node2 -o test_3.log run_test.sh c1 Baseline-v1 --caps
sbatch -w cluster-node3 -o test_4.log run_test.sh c1 Enhanced-v0
sbatch -w cluster-node3 -o test_5.log run_test.sh c1 Enhanced-v0 --caps
sbatch -w cluster-node4 -o test_6.log run_test.sh c1 Enhanced-v1
sbatch -w cluster-node4 -o test_7.log run_test.sh c1 Enhanced-v1 --caps
sbatch -w cluster-node5 -o test_8.log run_test.sh c1 Obstacle-v1
sbatch -w cluster-node5 -o test_9.log run_test.sh c1 Obstacle-v1 --caps
