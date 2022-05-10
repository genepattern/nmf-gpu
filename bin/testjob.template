#!/bin/bash
# Begin LSF Directives
#BSUB -P TRN008
#BSUB -W 00:50
#BSUB -nnodes 1
#BSUB -alloc_flags gpumps
#BSUB -J NMF-nondebug
#BSUB -o NMF.out.%J
#BSUB -e NMF.err.%J

export WORKINGDIR=__path_to_working_dir__
export BIONMFGPUPATH=__path_to_repo__
export CONSENSUSPY=__path_to_consensus.py
export GPUPROGRAMPATH=${BIONMFGPUPATH}/bin/nmf_mgpu_fun.py
module purge
module load DefApps
module load gcc/7.5.0
module load cuda/11.0.3
module load python
source activate /gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv
export CUPY_CACHE_DIR="/gpfs/wolf/trn008/scratch/${USER}/.cupy/kernel_cache"
cd ${WORKINGDIR}
mkdir jobdir.${LSB_JOBID}
export NUMTASKS=1
export CUDA_VISIBLE_DEVICES=0
echo CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES
nvidia-smi
date
cd ${WORKINGDIR}
${CONSENSUSPY} --verbose --gpuprogrampath=${GPUPROGRAMPATH} --jobdir=${WORKINGDIR}/jobdir.${LSB_JOBID} --inputfile=${WORKINGDIR}/ALL_AML_data.gct --mink=2 --maxk=2 --startseed=1 --seeds=10 --consecutive=40 --maxiterations=2000 --outputfileprefix=ALL_AML_data --interval=10 --keepintermediatefiles --numtasks=${NUMTASKS}
date
