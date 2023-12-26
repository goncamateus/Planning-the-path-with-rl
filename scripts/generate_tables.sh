#!/bin/bash

sbatch -w cluster-node1 -o test_0.log run_test.sh c1 --env baseline --setup pure
sbatch -w cluster-node1 -o test_1.log run_test.sh c1 --env baseline --setup frameskip
sbatch -w cluster-node2 -o test_2.log run_test.sh c1 --env baseline --setup caps
sbatch -w cluster-node2 -o test_3.log run_test.sh c1 --env baseline --setup fscaps
sbatch -w cluster-node3 -o test_4.log run_test.sh c1 --env enhanced --setup pure
sbatch -w cluster-node3 -o test_5.log run_test.sh c1 --env enhanced --setup frameskip
sbatch -w cluster-node4 -o test_6.log run_test.sh c1 --env enhanced --setup caps
sbatch -w cluster-node4 -o test_7.log run_test.sh c1 --env enhanced --setup fscaps
sbatch -w cluster-node4 -o test_8.log run_test.sh c1 --env obstacle --setup pure
sbatch -w cluster-node4 -o test_9.log run_test.sh c1 --env obstacle --setup frameskip
sbatch -w cluster-node5 -o test_10.log run_test.sh c1 --env obstacle --setup caps
sbatch -w cluster-node5 -o test_11.log run_test.sh c1 --env obstacle --setup fscaps
