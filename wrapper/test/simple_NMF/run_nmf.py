#!/gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv/bin/python3
# https://www.pnas.org/doi/10.1073/pnas.0308531101?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed
# https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-015-0485-4.pdf
# CUDA Hook Library: Failed to find symbol mem_find_dreg_entries, /gpfs/wolf/trn00
# 8/proj-shared/teammesirov/conda_envs/cupyenv/bin/python3: undefined symbol: __PA
# MI_Invalidate_region
# https://github.com/olcf/olcf-user-docs/issues/78
# test/bionmf.input.txt  -k {k}  -j 10  -t 40  -i 2000 -s {seed}
# sys.argv[1] path to txt array
# -k k factor
# -j how many iterations between convergence checks
# -t how many iterations with unchanging classification for convergence
# -i max iterations
# -s prng seed

import argparse
import sys
import cupy as cp
import os.path
from mpi4py import MPI

sys.path.append("/gpfs/wolf/trn008/scratch/liefeld/nmf-gpu/wrapper")
from readgct import NP_GCT

sys.path.append("/gpfs/wolf/trn008/scratch/liefeld/nmf-gpu/bin")
from nmf_mgpu_fun import runnmf


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  numtasks = comm.Get_size()
  print(f'rank: ({rank}), numtasks: ({numtasks})\n')
  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
  parser.add_argument('-m', '--inputmatrix', dest='inputmatrix', action='store')
  parser.add_argument('-k', '--kfactor', dest='kfactor', action='store')
  parser.add_argument('-j', '--checkinterval', dest='checkinterval', action='store')
  parser.add_argument('-t', '--threshold', dest='threshold', action='store')
  parser.add_argument('-i', '--maxiterations', dest='maxiterations', action='store')
  parser.add_argument('-s', '--seed', dest='seed', action='store')
  parser.add_argument('-o', '--outputprefix', dest='outputprefix', action='store')
  args = parser.parse_args()
  checkinterval = int(args.checkinterval)
  maxiterations = int(args.maxiterations)
  threshold = int(args.threshold)
  debug = args.verbose
  seed = int(args.seed)
  kfactor = int(args.kfactor)

  # V = cp.loadtxt(fname=args.inputmatrix)
  gct_data = NP_GCT(args.inputmatrix)
  V = gct_data.data
  print(str(V))
  print("Seed = " + str(args.seed))
  print("Verbose = "+str(debug))
  print(" args = " + str(args))

  RET_VAL = runnmf(inputmatrix=V, kfactor=kfactor, checkinterval=checkinterval, threshold=threshold, maxiterations=maxiterations, seed=seed, debug=debug)
  print(str(RET_VAL))
  #cp.savetxt(os.path.basename(args.inputmatrix) + '_H.txt', H)
  H_gct = NP_GCT(data=RET_VAL[0], rowNames=gct_data.rownames, rowDescrip=gct_data.rowdescriptions)
  W_gct = NP_GCT(data=RET_VAL[1], colNames=gct_data.columnnames)
  H_gct.write_gct(args.outputprefix+"_H.gct")
  W_gct.write_gct(args.outputprefix+"_W.gct")
  

  mpi4py.Finalize()
