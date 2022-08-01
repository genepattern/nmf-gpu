#!/usr/bin/bash
. /expanse/projects/mesirovlab/genepattern/modules/nmf-gpu-v1/env
export CONSENSUSPY=/expanse/projects/mesirovlab/genepattern/modules/nmf-gpu-v1/nmf-gpu/bin/consensus.unified.py
#export CONSENSUSPY=/expanse/lustre/projects/ddp242/kenneth/pure/nmf-gpu/bin/c.py
mpiexec -np ${SLURM_NTASKS} ${CONSENSUSPY} $@
