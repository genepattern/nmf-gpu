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


def runnmf(inputmatrix=None,kfactor=2,checkinterval=10,threshold=40,maxiterations=2000,seed=1,debug=False):
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
  #H = cp.random.rand(kfactor,M)
  H = cp.random.rand(kfactor,M) + 0.1
  #Hones = cp.ones((kfactor,M))
  #H = cp.add(H,Hones)
  # is it better to create the transposed matrix from the start?
  Ht = H.transpose()
  #W = cp.random.rand(N,kfactor)
  W = cp.random.rand(N,kfactor) + 0.1
  #Wones = cp.ones((N,kfactor))
  #W = cp.add(W,Wones)
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
    
    #olddebug = debug
    #debug = True
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
      # H is kxM matrix.  classification is 1xM array, with each element
      # the row index of the max of the elements in that column.
      # from the C code comments:
      #/*
      # * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
      # * That is, returns d_Idx[i], such that:
      # *      d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
      # *
      # * size_of( d_Idx ) >= height
      # * width <= pitch <= maxThreadsPerBlock
      # * In addition, "pitch" must be a multiple of 'memory_alignment'.
      # *
      # * Returns EXIT_SUCCESS or EXIT_FAILURE.
      # */

      if debug:
        print('checking classification...\n')
      #newclassification = cp.ndarray((M), dtype=cp.int16)
      #for thiscolumn in range(M):
      #  for thisrow in range(kfactor):
      #    if thisrow == 0:
      #      maxrow = 0
      #    else:
      #      # tie goes to incumbent?
      #      #if Ht[thisrow,thiscolumn] > Ht[thisrow,maxrow]:
      #      if H[thisrow,thiscolumn] > H[maxrow,thiscolumn]:
      #        maxrow = thisrow
      #  newclassification[thiscolumn] = maxrow
      newclassification = cp.argmax(H, axis=0)
      if debug:
        print(f'type(oldclassification): ({type(oldclassification)})\n')
        print(f'newclassification: ({newclassification})\n')
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
        if debug:
          if type(oldclassification) == type(newclassification):
            print(f'2. equal?: ({cp.array_equal(oldclassification, newclassification)})\n')
          else:
            print(f'2. type(oldclassification) != type(newclassification)\n')
        oldclassification = newclassification
        sameclassificationcount = 0
    iterationcount = iterationcount + 1
    #debug = olddebug
  if debug:
    print(f'iterationcount ({iterationcount})\n')
  # if we are here, we failed to return before we hit iteration limit
  # should raise an exception

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
