#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=prac3

# use our reservation
#SBATCH --reservation=cuda2025
#SBATCH --mem=16834

module purge
module load CUDA

make clean
make

./laplace3d
# ./laplace3d_new

# ncu laplace3d
# ncu laplace3d_new

# ncu --metrics "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum" laplace3d
# ncu --metrics "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum" laplace3d_new