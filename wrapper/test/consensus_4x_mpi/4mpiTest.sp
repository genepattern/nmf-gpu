#!/bin/bash
# Begin LSF Directives
#BSUB -P TRN008
#BSUB -W 00:50
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J NMF-nondebug
#BSUB -o MPI.out.%J
#BSUB -e MPI.err.%J

export TEST_ROOT=${PWD}
export SRC_ROOT="/gpfs/wolf/trn008/scratch/${USER}/nmf-gpu/bin"
#export SRC_ROOT=${TEST_ROOT}
export PYVENVPATH=/gpfs/wolf/trn008/proj-shared/teammesirov/pyvenv
module load gcc
module load DefApps
module load cuda
module load python
source activate /gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv
export CUPY_CACHE_DIR="/gpfs/wolf/trn008/scratch/${USER}/.cupy/kernel_cache"

mkdir MPI.jobdir.${LSB_JOBID}
cd  MPI.jobdir.${LSB_JOBID}
export WORKINGDIR=${PWD}

#export NUMTASKS=1
#export CUDA_VISIBLE_DEVICES=0
export NUMTASKS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

#echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
nvidia-smi
date
jsrun --smpiargs="-gpu" --nrs=${NUMTASKS} --tasks_per_rs=1 --cpu_per_rs=1 --gpu_per_rs=1 --rs_per_host=${NUMTASKS} --bind=rs  python3 $SRC_ROOT/consensus.pure.py --inputfile /gpfs/wolf/trn008/scratch/liefeld/nmf-gpu/test_data/ALL_AML_data.gct --outputfileprefix foo --mink 2 --maxk 9 --interval 10 --consecutive 40 --maxiterations 2000 --jobdir ${PWD} --startseed 1 --seeds 100 -v 
date
