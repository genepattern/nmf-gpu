#!/usr/bin/bash
. /expanse/projects/mesirovlab/genepattern/modules/nmf-gpu-v3/env


# singularity shell --nv -B /expanse/projects/mesirovlab/genepattern/modules/singularity_version:/expanse/projects/mesirovlab/genepattern/modules/singularity_version ./nmf-gpu.rapids.img

#
echo Job starting at `date` in `pwd`

export JOBDIR=`pwd`

#singularity exec --nv -B /expanse/projects/mesirovlab/genepattern/modules/:/expanse/projects/mesirovlab/genepattern/modules/ /expanse/projects/mesirovlab/genepattern/modules/singularity_version/nmf-gpu.rapids.img  mpiexec -n 2 python3 /expanse/projects/mesirovlab/genepattern/modules/singularity_version/nmf-gpu/bin/consensus.unified.py  --parastrategy=inputmatrix --jobdir=${JOBDIR} --inputfile=/expanse/projects/mesirovlab/genepattern/modules/BRCA_DESeq2_normalized_19309x40.preprocessed.gct --mink=2 --maxk=5 --startseed=5 --seeds=100 --consecutive=40 --maxiterations=2000 --outputfileprefix=ALL_AML_data --interval=10 --numtasks=${SLURM_NTASKS} --inputfiletype=gct --klerrordiffmax=5.0 --outputfiletype=gct 

singularity exec --nv -B /expanse/projects/mesirovlab/genepattern/:/expanse/projects/mesirovlab/genepattern/ /expanse/projects/mesirovlab/genepattern/modules/nmf-gpu-v3/nmf-gpu.rapids.img  mpiexec -n ${SLURM_NTASKS} python3 /expanse/projects/mesirovlab/genepattern/modules/nmf-gpu-v3/nmf-gpu/bin/consensus.unified.py $@ 

