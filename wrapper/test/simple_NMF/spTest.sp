#!/bin/bash
# Begin LSF Directives
#BSUB -P TRN008
#BSUB -W 00:50
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J NMF-nondebug
#BSUB -o NMF.out.%J
#BSUB -e NMF.err.%J

export TEST_ROOT=${PWD}
export PYVENVPATH=/gpfs/wolf/trn008/proj-shared/teammesirov/pyvenv
module load gcc
module load DefApps
module load cuda
module load python
source activate /gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv
export CUPY_CACHE_DIR="/gpfs/wolf/trn008/scratch/${USER}/.cupy/kernel_cache"

mkdir NMF.jobdir.${LSB_JOBID}
cd  NMF.jobdir.${LSB_JOBID}
export WORKINGDIR=${PWD}

export NUMTASKS=1
export CUDA_VISIBLE_DEVICES=0
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
nvidia-smi
date
jsrun --smpiargs="-gpu" --nrs=${NUMTASKS} --tasks_per_rs=1 --cpu_per_rs=1 --gpu_per_rs=1 --rs_per_host=${NUMTASKS} --bind=rs  python3 $TEST_ROOT/run_nmf.py -m /gpfs/wolf/trn008/scratch/liefeld/nmf-gpu/test_data/ALL_AML_data.gct -o foo -k 2 -j 10 -t 40 -i 2000 -s 1 -v  
date
