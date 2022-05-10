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


def runnmf(inputmatrix=None,kfactor=2,checkinterval=10,threshold=40,maxiterations=2000,seed=1,verbose=False):
  # read input file, create array on device
  #V = cp.loadtxt(fname=args.inputmatrix)
  V = inputmatrix
  if debug:
    print(f'inputarray.shape: ({V.shape})\n')
    print(f'V: ({V})\n')
  
  # seed the PRNG
  #cp.random.seed(int(args.seed))
  cp.random.seed(seed)
  
  if debug:
    print('start of nmf_mgpu.py\n')
    
  N = V.shape[0]
  M = V.shape[1]
  #kfactor = int(args.kfactor)
  #kfactor = kfactor
  # create H and W random on device
  # H = M (inputarray.shape[1]) x k (stored transposed), W = N (inputarra.shape[0]x k
  # in the bionmf-gpu code, 
  # set_random_values( p_dH, nrows, K, Kp,
  # set_random_values( d_W, N, K, Kp,
  # should be transposed for H...
  # this is over interval [0, 1), might need to add a smidge...
  H = cp.random.rand(kfactor,M)
  # is it better to create the transposed matrix from the start?
  Ht = H.transpose()
  W = cp.random.rand(N,kfactor)
  Wt = W.transpose()
  if debug:
    print(f'initial H: ({H})\n')
    print(f'initial Ht: ({Ht})\n')
    print(f'initial W: ({W})\n')
    print(f'initial Wt: ({Wt})\n')
  
  iterationcount = 0
  oldclassification = None
  sameclassificationcount = 0
  while iterationcount < maxiterations:
  
    # update Ht
    # * WH(N,BLMp) = W * pH(BLM,Kp)
    # * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
    # * Haux(BLM,Kp) = W' * WH(N,BLMp)
    # * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
    
    # WH = W * H
    # WH (N x M) = (N x k) * (k x M)
    if debug:
      print(f' W.shape: {W.shape}, H.shape: {H.shape}\n')
    WH = cp.matmul(W, H)
    if debug:
      print(f'WH: ({WH})\n')
  
    # AUX = V (input matrix) ./ (W*H)
    # AUX (N x M)
    #AUX = cp.divide(V, WH)
    # overwrite WH, I wonder if that breaks anything...
    WH = cp.divide(V, WH)
    #print(f'AUX: ({AUX})\n')
    if debug:
      print(f'WH: ({WH})\n')
    
    # WTAux = Wt * AUX
    if debug:
      print(f'Wt: ({Wt})\n')
    WTAUX = cp.matmul(Wt, WH, dtype=cp.float64)
    if debug:
      print(f' WTAUX.shape: {WTAUX.shape}\n')
      print(f'WTAUX: ({WTAUX})\n')
    
    # how do we get reduced an accumulated ACCWT below?
    # sum each column down to a single value...
    ACCW = cp.sum(W, axis=0)
    
    # WTAUXDIV = WTAUX ./ ACCWT
    
    if debug:
      print(f' WTAUX.shape: {WTAUX.shape}, ACCW.shape: {ACCW.shape}\n')
      print(f'ACCW: ({ACCW})\n')
    WTAUXDIV = cp.divide(WTAUX.transpose(), ACCW)
    WTAUXDIV = WTAUXDIV.transpose()
    if debug:
      print(f'WTAUXDIV: ({WTAUXDIV})\n')
    
    # H = H .* WTAUXDIV
    Hnew = cp.multiply(H, WTAUXDIV)
    if debug:
      print(f'Hnew: ({Hnew})\n')
    
    if debug:
      print('before update W\n')
    # update W
    # * WH(BLN,Mp) = W(BLN,Kp) * H
    # * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
    # * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    # generate ACCUMH
    ACCH = cp.sum(H, axis=1)
    if debug:
      print(f'H: ({H})\n')
      print(f'ACCH: ({ACCH})\n')
    
    # skip steps up to AUX
    # from update_W notes:
    #  * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # should I be using the original Ht here?
    HTAUX = cp.matmul(WH, Ht)
    if debug:
      print(f'HTAUX: ({HTAUX})\n')
    
    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    WWAUX = cp.multiply(W, HTAUX)
    if debug:
      print(f'WWAUX: ({WWAUX})\n')
    
    Wnew = cp.divide(WWAUX, ACCH)
    if debug:
      print(f'Wnew: ({Wnew})\n')
    H = Hnew
    Ht = H.transpose()
    W = Wnew
    Wt = W.transpose()
    
    if debug:
      print('after update W\n')
      print(f'H: ({H})\n')
      print(f'W: ({W})\n')
  
    # check classification for Ht
    # * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
    # * That is, returns d_Idx[i], such that:
    # *      d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
    # *
    # * size_of( d_Idx ) >= height
    # * width <= pitch <= maxThreadsPerBlock
    # * In addition, "pitch" must be a multiple of 'memory_alignment'.
    # Ht is M x k, need array M x 1 to store column index
   
    if iterationcount > checkinterval and divmod(iterationcount, checkinterval)[1] == 0:
      # check classification
      if debug:
        print('checking classification...\n')
      newclassification = cp.ndarray((M, 1), dtype=cp.int16)
      for thisrow in range(M):
        for thiscolumn in range(kfactor):
          if thiscolumn == 0:
            maxrow = 0
          else:
            # tie goes to incumbent?
            if Ht[thisrow,thiscolumn] > maxrow:
              maxrow = thisrow
        newclassification[thisrow] = maxrow
      if debug:
        print(f'type(oldclassification): ({type(oldclassification)})\n')
      if type(oldclassification) == type(newclassification) and cp.array_equal(oldclassification, newclassification) == True:
        if debug:
          print(f'1. equal?: ({cp.array_equal(oldclassification, newclassification)})\n')
        sameclassificationcount = sameclassificationcount + 1
        if sameclassificationcount >= threshold:
          if debug:
            print(f'classification unchanged in {sameclassificationcount} trials,breaking.\n')
            print(f'H: ({H})\n')
          #cp.savetxt(os.path.basename(args.inputmatrix) + '_H.txt', H)
          #break
          return((W,H))
      else:
        oldclassification = newclassification
        sameclassificationcount = 0
        if iterationcount > 0:
          if debug:
            print(f'2. equal?: ({cp.array_equal(oldclassification, newclassification)})\n')
    iterationcount = iterationcount + 1
  if debug:
    print(f'iterationcount ({iterationcount})\n')

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
  args = parser.parse_args()
  checkinterval = int(args.checkinterval)
  maxiterations = int(args.maxiterations)
  threshold = int(args.threshold)
  debug = args.verbose
  V = cp.loadtxt(fname=args.inputmatrix)
  W, H = runnmf(inputmatrix=V, kfactor=int(args.kfactor), checkinterval=checkinterval, threshold=threshold, maxiterations=maxiterations, seed=int(args.seed), verbose=debug)
  cp.savetxt(os.path.basename(args.inputmatrix) + '_H.txt', H)
  mpi4py.Finalize()
