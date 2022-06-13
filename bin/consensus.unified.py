#!/expanse/lustre/projects/ddp242/kenneth/pure/install/venv/bin/python3
# from NMF GPU Functional specification document:
# ...
#A pseudocode representation of the workflow we’re looking for is below. We are also interested in timing statistics - most important is wall clock time from start to finish, but if we can time each iteration of k that would be very helpful also.
#
#
#For each value of k from 2 to 9
#        Decide on 100 integer values to serve as random seeds
#For each i in 100 runs of NMF
#        Initialize W and H according to the ith random seed
#        Launch NMF with the given value of k, other parameters at their default values
#Assign samples to clusters according to the derived H matrix with the following procedure:
#Initialize an empty list C of length equal to number of columns in H
#For each column j in the H matrix:
#Find the row index m of the maximal value in column j
#Set C[ j ] = m
#Compute and output consensus matrix
#Compute cophenetic correlation coefficient (CCC)
#Output graph of CCC vs. k
#Deliverables
#
#
#1. Implement a random seed parameter as an input to the NMF algorithm. This would be an optional parameter which, when passed, would serve as the random seed for the initialization of the H and W matrices.
#2. Implement a consensus matrix as an output. The script that iterates over various (e.g. 100) initializations would create a consensus matrix of samples vs. samples where the value of matrix cell (i,j) is the frequency with which sample i is in the same cluster as sample j. The PNAS paper referenced above provides examples of consensus matrix output.
#3. Implement the cophenetic correlation coefficient as a summary statistic. The CCC is a measure of the goodness of fit of a given clustering. It is available in the scikit Python library.
#4. Run the algorithm above over the dataset. As described above, the script would iterate over several values of k and several initializations using different random seeds, compute the consensus matrix and CCC for each value of k, and output the consensus matrices and a graph of CCC versus k.

# https://mpi4py.readthedocs.io/en/stable/tutorial.html#mpi-io
# https://docs.h5py.org/en/stable/mpi.html
# https://www.pnas.org/doi/10.1073/pnas.0308531101?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed
# https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-015-0485-4.pdf
# CUDA Hook Library: Failed to find symbol mem_find_dreg_entries, /gpfs/wolf/trn00
# 8/proj-shared/teammesirov/conda_envs/cupyenv/bin/python3: undefined symbol: __PA
# MI_Invalidate_region
# https://github.com/olcf/olcf-user-docs/issues/78
# https://mpi4py.readthedocs.io/en/stable/tutorial.html#cuda-aware-mpi-python-gpu-arrays
# https://mpi4py.readthedocs.io/en/stable/reference/mpi4py.MPI.Comm.html#mpi4py. MPI.Comm.Allgatherv
# https://buildmedia.readthedocs.org/media/pdf/mpi4py/latest/mpi4py.pdf
# https://github.com/scikit-learn/scikit-learn/blob/5f3d1e57a91e7c89fe3485971188f1ecd335f2c1/sklearn/decomposition/_nmf.py#L856

# set a hundred seeds (could also read from a file, range with step, etc.):
import os
import os.path
import sys
import numpy
import shutil
import argparse
import subprocess
import shlex
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import gp.data
import pandas
import traceback
import cupy as cp
import time
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial.distance
from sklearn import cluster
import heapq
import pandas as pd
import types

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.rank
mpi_size = comm.size
#rank = mpi_rank
#numtasks = mpi_size


def divide_almost_equally(arr, num_chunks):
    arr = sorted(arr, reverse=True)
    heap = [(0, idx) for idx in range(num_chunks)]
    heapq.heapify(heap)
    sets = {}
    for i in range(num_chunks):
        sets[i] = []
    arr_idx = 0
    while arr_idx < len(arr):
        set_sum, set_idx = heapq.heappop(heap)
        sets[set_idx].append(arr[arr_idx])
        set_sum += arr[arr_idx]
        heapq.heappush(heap, (set_sum, set_idx))
        arr_idx += 1
    return list(sets.values())

def runnmf(inputmatrix=None,kfactor=2,checkinterval=10,threshold=40,maxiterations=2000,seed=1,debug=False, comm=None, parastrategy='serial', klerrordiffmax=None):
  EPSILON = cp.finfo(cp.float32).eps
  olddebug = debug
  # read input file, create array on device
  #V = cp.loadtxt(fname=args.inputmatrix)
  # for really big inputs, should probably feed in N and M as args,
  # then each task read in only its columns and rows from the input
  # matrix file.

  if comm == None or parastrategy in ('serial', 'kfactor'):
    rank = 0
    numtasks = 1
  else:
    rank = comm.Get_rank()
    numtasks = comm.Get_size()
  V = inputmatrix
  if debug:
    print(f'{rank}: inputarray.shape: ({V.shape})\n')
    print(f'{rank}: V: ({V})\n')

  # seed the PRNG
  cp.random.seed(seed)

  N = V.shape[0]
  M = V.shape[1]
  # padding?
  # if cols = 9, numtasks = 2, colspertask = 4, padding = 1
  # rank 0:
  # mystartcol = 0, myendcol = 4, numcols = 5
  # rank 1:
  # mystartcol = 5, myendcol = 9, numcols = 4
  colspertask, colremainder = divmod(M, numtasks)
  if colremainder == 0:
    colpad = 0
    mystartcol = colspertask * rank
    myendcol = colspertask * (rank + 1) - 1
  else:
    colpad = (colspertask + 1) * numtasks - M
    mystartcol = (colspertask + 1) * rank
    if rank == numtasks - 1:
      # I'm the last
      myendcol = mystartcol + (colspertask + 1) - colpad - 1
    else:
      myendcol = mystartcol + (colspertask + 1) - 1
  Hsendcountlist = []
  for tn in range(numtasks):
    if colremainder == 0:
      Hsendcountlist.append(kfactor * colspertask)
    else:
      if tn == numtasks - 1:
        #last task:
        Hsendcountlist.append(kfactor * ((colspertask + 1) - colpad))
      else:
        Hsendcountlist.append(kfactor * (colspertask + 1))
  rowspertask, rowremainder = divmod(N, numtasks)
  if rowremainder == 0:
    rowpad = 0
    mystartrow = rowspertask * rank
    myendrow = rowspertask * (rank + 1) - 1
  else:
    rowpad = (rowspertask + 1) * numtasks - N
    mystartrow = (rowspertask + 1) * rank
    if rank == numtasks - 1:
      # I'm the last
      myendrow = mystartrow + (rowspertask + 1) - rowpad - 1
    else:
      myendrow = mystartrow + (rowspertask + 1) - 1
  Wsendcountlist = []
  for tn in range(numtasks):
    if rowremainder == 0:
      Wsendcountlist.append(kfactor * rowspertask)
    else:
      if tn == numtasks - 1:
        #last task:
        Wsendcountlist.append(kfactor * ((rowspertask + 1) - rowpad))
      else:
        Wsendcountlist.append(kfactor * (rowspertask + 1))
  myrowcount = (myendrow + 1) - mystartrow
  mycolcount = (myendcol + 1) - mystartcol
  myVcols = V[:,mystartcol:myendcol + 1]
  myVrows = V[mystartrow:myendrow + 1,:]
  if debug:
    print(f'{rank}: mystartrow: {mystartrow}, myendrow: {myendrow}, mystartcol: {mystartcol}, myendcol: {myendcol}, myrowcount: {myrowcount}, mycolcount: {mycolcount}, Hsendcountlist: {Hsendcountlist}, Wsendcountlist: {Wsendcountlist}\n')
  # do I need to add the padded rows and columns to the last task?
  # or are the matrix operations okay?

  # create H and W random on device
  # H = M (inputarray.shape[1]) x k (stored transposed), W = N (inputarra.shape[0]x k
  # in the bionmf-gpu code,
  # set_random_values( p_dH, nrows, K, Kp,
  # set_random_values( d_W, N, K, Kp,
  # should be transposed for H...
  # Is there any danger of H and W generated from rand()
  # getting out of sync amongst the tasks?
  # this is over interval [0, 1), might need to add a smidge...
  H = cp.random.rand(kfactor,M) + 0.001
  # is it better to create the transposed matrix from the start?
  Ht = H.transpose()
  W = cp.random.rand(N,kfactor) + 0.001
  Wt = W.transpose()
  if debug:
    print(f'{rank}: initial H: ({H})\n')
    print(f'{rank}: initial Ht: ({Ht})\n')
    print(f'{rank}: initial W: ({W})\n')
    print(f'{rank}: initial Wt: ({Wt})\n')
    print(f'{rank}: V: ({V})\n')
    print(f'{rank}: WH: ({cp.matmul(W,H)})\n')
    # write out pickled W and H
    #WFO = open(f'k-{kfactor}.seed-{seed}.initialW.pkl', 'w')
    #pickle.dump(cp.asnumpy(W), WFO, protocol=0)
    #WFO.close()
    #HFO = open(f'k-{kfactor}.seed-{seed}.initialH.pkl', 'w')
    #pickle.dump(cp.asnumpy(H), HFO, protocol=0)
    #HFO.close()
    cp.asnumpy(W).dump(f'k-{kfactor}.seed-{seed}.initialW.pkl')
    cp.asnumpy(H).dump(f'k-{kfactor}.seed-{seed}.initialH.pkl')

  iterationcount = 0
  oldclassification = None
  sameclassificationcount = 0
  oldserialerror = None
  oldmpierror = None
  #KLFO = open('kldiv.tsv', 'w')
  while iterationcount < maxiterations:

    # update Ht
    # * WH(N,BLMp) = W * pH(BLM,Kp)
    # * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
    # * Haux(BLM,Kp) = W' * WH(N,BLMp)
    # * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W

    # WH = W * H
    # WH (N x M) = (N x k) * (k x M)
    if debug:
      print(f'{rank}:  W.shape: {W.shape}, H.shape: {H.shape}\n')
      print(f'{rank}: W.shape {W.shape} H[:,mystartcol:myendcol + 1].shape {H[:,mystartcol:myendcol + 1].shape}, mystartcol {mystartcol}, myendcol: {myendcol}\n')
      print(f'{rank}: H: {H}\n')
      print(f'{rank}: H[:,mystartcol:myendcol + 1]: {H[:,mystartcol:myendcol + 1]}\n')
    WHm = cp.matmul(W, H[:,mystartcol:myendcol + 1])
    if debug:
      print(f'{rank}: after matmul, WH.shape: {WHm.shape}\n')
      print(f'{rank}: WHm: ({WHm})\n')
      print(f'{rank}: update H, matmul(W, H[:,mystartcol:myendcol + 1]), WHm: ({WHm})\n')

    # AUX = V (input matrix) ./ (W*H)
    # AUX (N x M)
    #AUX = cp.divide(V, WH)
    WH = cp.divide(V[:,mystartcol:myendcol + 1], WHm)
    if debug:
      print(f'{rank}: after divide, WH.shape: {WH.shape}\n')
      print(f'{rank}: update H, divide(V[:,mystartcol:myendcol + 1], WHm), WH: ({WH})\n')

    # WTAux = Wt * AUX
    if debug:
      print(f'{rank}: Wt: ({Wt})\n')
    WTAUX = cp.matmul(Wt, WH)
    if debug:
      print(f'{rank}: Wt.shape {Wt.shape} WH.shape {WH.shape} WTAUX.shape {WTAUX.shape}\n')
      print(f'{rank}:  WTAUX.shape: {WTAUX.shape}\n')
      print(f'{rank}: WTAUX: ({WTAUX})\n')
      print(f'{rank}: update H, matmul(Wt, WH), WTAUX: ({WTAUX})\n')

    # how do we get reduced an accumulated ACCWT below?
    # sum each column down to a single value...
    ACCW = cp.sum(W, axis=0)

    # WTAUXDIV = WTAUX ./ ACCWT

    if debug:
      print(f'{rank}:  WTAUX.shape: {WTAUX.shape}, ACCW.shape: {ACCW.shape}\n')
      print(f'{rank}: ACCW: ({ACCW})\n')
    WTAUXDIV = cp.divide(WTAUX.transpose(), ACCW)
    WTAUXDIV = WTAUXDIV.transpose()
    if debug:
      print(f'{rank}: WTAUXDIV: ({WTAUXDIV})\n')

    # H = H .* WTAUXDIV
    Hnew = cp.multiply(H[:,mystartcol:myendcol + 1], WTAUXDIV)
    if debug:
      print(f'{rank}: Hnew: ({Hnew}, Hnew.shape {Hnew.shape}, H[:,mystartcol:myendcol + 1].shape {H[:,mystartcol:myendcol + 1].shape}, WTAUXDIV.shape {WTAUXDIV.shape})\n')
      print(f'{rank}: Hnew: ({Hnew})\n')
    # sync H to all devices
    if parastrategy == 'inputmatrix':
      # Daniel's code suggests flattening before the allgather and
      # reshaping after.  Also, for recvbuf, use a tuple of receive
      # array and sendcount
      # looks like the old code, without ravel and sendcount was
      # trying to send into the first row until it overran it...
      Hshape = H.shape
      # ravel of subset of columns will put columns of the next row
      # next to each other, which will change order elements when
      # reshape happens.  Need to add columns for H and add rows for W...
      Hnewflat = Hnew.ravel(order='F')
      Hrecv = cp.empty(H.size)
      if debug:
        print(f'{rank}: Hnew {Hnew}\n')
        print(f'{rank}: Hnewflat {Hnewflat}\n')
        print(f'{rank}: H.shape {H.shape}\n')
        print(f'{rank}: Hrecv.shape {Hrecv.shape}\n')
      cp.cuda.Stream.null.synchronize()
      comm.Allgatherv(sendbuf=Hnewflat,recvbuf=(Hrecv,Hsendcountlist))
      if debug:
        print(f'{rank}: after Allgatherv, Hrecv.shape {Hrecv.shape}\n')
        print(f'{rank}: after Allgatherv, Hrecv {Hrecv}\n')
      Hnewfull = Hrecv.reshape(Hshape, order='F')
      if debug:
        print(f'{rank}: after Allgatherv, reshape, H.shape {H.shape}\n')
        print(f'{rank}: after Allgatherv, reshape, H {H}\n')
      H = Hnewfull
    else:
      cp.cuda.Stream.null.synchronize()
      H = Hnew
    Ht = H.transpose()

    if debug:
      print(f'{rank}: before update W\n')
    # update W
    # * WH(BLN,Mp) = W(BLN,Kp) * H
    # * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
    # * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    # generate ACCUMH
    ACCH = cp.sum(H, axis=1)
    if debug:
      print(f'{rank}: H: ({H})\n')
      print(f'{rank}: ACCH: ({ACCH})\n')

    WHm = cp.matmul(W[mystartrow:myendrow + 1,:], H)
    WH = cp.divide(V[mystartrow:myendrow + 1,:], WHm)
    # from update_W notes:
    #  * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # should I be using the original Ht here?
    HTAUX = cp.matmul(WH, Ht)
    if debug:
      print(f'{rank}: HTAUX: ({HTAUX})\n')

    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    WWAUX = cp.multiply(W[mystartrow:myendrow + 1,:], HTAUX)
    if debug:
      print(f'{rank}: WWAUX: ({WWAUX})\n')

    olddebug = debug
    Wnew = cp.divide(WWAUX, ACCH)
    if debug:
      print(f'{rank}: Wnew: ({Wnew})\n')
      print(f'{rank}: Hnew: ({Hnew}), Hnew.shape: {Hnew.shape}\n')
      print(f'{rank}: Wnew: ({Wnew}), Wnew.shape: {Wnew.shape}\n')
      print(f'{rank}: Wnew: ({Wnew})\n')
    # sync W to all devices
    if parastrategy == 'inputmatrix':
      Wshape = W.shape
      Wnewflat = Wnew.ravel(order='C')
      Wrecv = cp.empty(W.size)
      if debug:
        print(f'{rank}: Wnew {Wnew}\n')
        print(f'{rank}: Wnewflat {Wnewflat}\n')
        print(f'{rank}: W.shape {W.shape}\n')
        print(f'{rank}: Wrecv.shape {Wrecv.shape}\n')
      cp.cuda.Stream.null.synchronize()
      comm.Allgatherv(sendbuf=Wnewflat,recvbuf=(Wrecv, Wsendcountlist))
      Wnewfull = Wrecv.reshape(Wshape, order='C')
      W = Wnewfull
    else:
      cp.cuda.Stream.null.synchronize()
      H = Hnew
      W = Wnew

    if debug:
      print(f'{rank}: after Allgatherv, W.shape {W.shape}\n')
      print(f'{rank}: after Allgatherv, W {W}\n')
    Wt = W.transpose()

    if debug:
      print(f'{rank}: after update W\n')
      print(f'{rank}: H: ({H}), H.shape: {H.shape}\n')
      print(f'{rank}: W: ({W}), W.shape: {W.shape}\n')
      print(f'{rank}, {iterationcount}: after sync, V: ({V})\n')
      print(f'{rank}, {iterationcount}: after sync, W: ({W})\n')
      print(f'{rank}, {iterationcount}: after sync, H: ({H})\n')
      print(f'{rank}, {iterationcount}: after sync, WH: ({cp.matmul(W,H)})\n')


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
      # check KL divergence
      if not klerrordiffmax == None:
        if debug:
          print(f'{rank}: checking KL divergence, since klerrordiffmax: ({klerrordiffmax})\n')
        if parastrategy == 'inputmatrix':
          if debug:
            print(f'{rank}: checking MPI KL divergence, since parastrategy: ({parastrategy})\n')
          WHkl = cp.dot(W,Hnew)
          if debug:
            print(f'{rank}: WHkl.shape ({WHkl.shape})\n')
          WH_datakl = WHkl.ravel()
          if debug:
            print(f'{rank}: V[:,mystartcol:myendcol + 1].shape {V[:,mystartcol:myendcol + 1].shape}\n')
          X_datakl = V[:,mystartcol:myendcol + 1].ravel()
          if debug:
            print(f'{rank}: WH_datakl: ({WH_datakl})\n')
            print(f'{rank}: X_datakl: ({X_datakl})\n')
          indices = X_datakl > EPSILON
          antiindices = X_datakl <= EPSILON
          if debug:
            print(f'{rank}: antiindices: ({antiindices})\n')
          WH_datakl = WH_datakl[indices]
          X_datakl = X_datakl[indices]
          WH_datakl[WH_datakl == 0] = EPSILON
          if debug:
            print(f'{rank}: after indices and EPSILON, WH_datakl: ({WH_datakl})\n')
            print(f'{rank}: after indices and EPSILON, X_datakl: ({X_datakl})\n')
          sum_WH = cp.dot(cp.sum(W, axis=0), cp.sum(Hnew, axis=1))
          div = X_datakl / WH_datakl
          if debug:
            print(f'{rank}: X_datakl / WH_datakl, div: ({div})\n')
          res = cp.dot(X_datakl, cp.log(div))
          if debug:
            print(f'{rank}: cp.log(div) ({cp.log(div)})\n')
            print(f'{rank}: dot(X_data, cp.log(div)), res: ({res})\n')
          res += sum_WH - X_datakl.sum()
          if debug:
            print(f'{rank}: adding sum_WH ({sum_WH}) - X_datakl.sum() ({X_datakl.sum()}) to starting res,  ending res: ({res})\n')
          totalres = cp.zeros_like(res)
          if debug:
            print(f'{rank}: empty totalres: ({totalres})\n')
          comm.Allreduce(res, totalres)
          error = numpy.sqrt(2 * totalres)
          if type(oldmpierror) == types.NoneType:
            errordiff = None
            oldmpierror = error
          else:
            errordiff = oldmpierror - error
            oldmpierror = error
          if debug:
            print(f'{rank}: afer Reduce, totalres: ({totalres})\n')
            print(f'{rank}: MPI KL divergence, error: ({error})\n')
            print(f'{rank}: mpi error difference: {errordiff}\n')
          if (not type(errordiff) == types.NoneType) and errordiff < klerrordiffmax:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) < klerrordiffmax ({klerrordiffmax}), return(W,H)\n')
            return(W,H)
          else:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) not less than klerrordiffmax ({klerrordiffmax})\n')
        else:
          if debug:
            print(f'{rank}: checking serial KL divergence, since parastrategy: ({parastrategy})\n')
          cp.cuda.Stream.null.synchronize()
          WHkl = cp.dot(W,H)
          cp.cuda.Stream.null.synchronize()
          WH_datakl = WHkl.ravel()
          cp.cuda.Stream.null.synchronize()
          X_datakl = V.ravel()
          cp.cuda.Stream.null.synchronize()
          indices = X_datakl > EPSILON
          cp.cuda.Stream.null.synchronize()
          WH_datakl = WH_datakl[indices]
          cp.cuda.Stream.null.synchronize()
          X_datakl = X_datakl[indices]
          cp.cuda.Stream.null.synchronize()
          WH_datakl[WH_datakl == 0] = EPSILON
          cp.cuda.Stream.null.synchronize()
          sum_WH = cp.dot(cp.sum(W, axis=0), cp.sum(H, axis=1))
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: cp.sum(W, axis=0).shape ({cp.sum(W, axis=0).shape}), cp.sum(H, axis=1).shape ({cp.sum(H, axis=1).shape})\n')
            print(f'{rank}: dot(cp.sum(W, axis=0), cp.sum(H, axis=1), sum_WH: ({sum_WH})\n')
          div = X_datakl / WH_datakl
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: X_datakl / WH_datakl, div: ({div})\n')
          res = cp.dot(X_datakl, cp.log(div))
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: cp.log(div) ({cp.log(div)})\n')
            print(f'{rank}: dot(X_data, cp.log(div)), res: ({res})\n')
            print(f'{rank}: adding sum_WH ({sum_WH}) - X_datakl.sum() ({X_datakl.sum()}) to starting res,  ending res: ({res})\n')
          res += sum_WH - X_datakl.sum()
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: KL divergence, serial calculation, res: ({res})\n')
          error = cp.sqrt(2 * res)
          if debug:
            print(f'{rank}: oldserialerror: {oldserialerror}\n')
          if type(oldserialerror) == types.NoneType:
            errordiff = None
            oldserialerror = error
          else:
            errordiff = oldserialerror - error
            if debug:
              print(f'{rank}: serial error difference: {errordiff}\n')
            oldserialerror = error
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: KL divergence, serial calculation, error: ({error})\n')
          #KLFO.write(f'{iterationcount}\t{error}\n')
          if (not type(errordiff) == types.NoneType) and errordiff < klerrordiffmax:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) < klerrordiffmax ({klerrordiffmax}), return(W,H)\n')
            return(W,H)
          else:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) not less than klerrordiffmax ({klerrordiffmax})\n')

      else:
        if debug:
          print(f'{rank}: checking classification change, since klerrordiffmax: ({klerrordiffmax})\n')
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
          if rank == 0:
            print(f'{rank}: checking classification...\n')
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
          if rank == 0:
            print(f'{rank}: type(oldclassification): ({type(oldclassification)})\n')
            print(f'{rank}: newclassification: ({newclassification})\n')
        if type(oldclassification) == type(newclassification) and cp.array_equal(oldclassification, newclassification) == True:
          if debug:
            if rank == 0:
              print(f'{rank}: 1. equal?: ({cp.array_equal(oldclassification, newclassification)})\n')
          sameclassificationcount = sameclassificationcount + 1
          if sameclassificationcount >= threshold:
            if debug:
              if rank == 0:
                print(f'{rank}: classification unchanged in {sameclassificationcount} trials,breaking.\n')
                print(f'{rank}: H: ({H})\n')
            #cp.savetxt(os.path.basename(args.inputmatrix) + '_H.txt', H)
            #break
            debug = olddebug
            if debug:
              print(f'{rank}: V: ({V})\n')
            return(W,H)
        else:
          if debug:
            if type(oldclassification) == type(newclassification):
              print(f'{rank}: 2. equal?: ({cp.array_equal(oldclassification, newclassification)})\n')
            else:
              print(f'{rank}: 2. type(oldclassification) != type(newclassification)\n')
          oldclassification = newclassification
          sameclassificationcount = 0
    iterationcount = iterationcount + 1
  #KLFO.close()
  if debug:
    print(f'{rank}: iterationcount ({iterationcount})\n')
  debug = olddebug



############  add local functions
sys.path.append("/expanse/lustre/projects/ddp242/kenneth/pure/nmf-gpu/wrapper")
from readgct import NP_GCT
#sys.path.append("/expanse/lustre/projects/ddp242/kenneth/pure/nmf-gpu/bin")
#from nmf_mgpu_mpi import runnmf
###########

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--startseed', dest='startseed', action='store')
parser.add_argument('-b', '--vref', dest='vref', action='store')
parser.add_argument('-c', '--nocleanup', dest='nocleanup', action='store_true')
parser.add_argument('-d', '--jobdir', dest='jobdir', action='store')
parser.add_argument('-e', '--maxk', dest='maxk', action='store')
parser.add_argument('-f', '--href', dest='href', action='store')
parser.add_argument('-g', '--noconsensus', dest='noconsensus', action='store_true')
parser.add_argument('-i', '--inputfile', dest='inputfile', action='store')
parser.add_argument('-j', '--interval', dest='interval', action='store')
parser.add_argument('-k', '--keepintermediatefiles', dest='keepintermediatefiles', action='store_true')
parser.add_argument('-l', '--klerrordiffmax', dest='klerrordiffmax', action='store')
parser.add_argument('-m', '--mink', dest='mink', action='store')
parser.add_argument('-n', '--numtasks', dest='mpitasks', action='store')
parser.add_argument('-o', '--outputfileprefix', dest='outputfileprefix', action='store')
parser.add_argument('-p', '--gpuprogrampath', dest='gpuprogrampath', action='store')
parser.add_argument('-r', '--seeds', dest='seeds', action='store')
parser.add_argument('-s', '--parastrategy', dest='parastrategy', action='store', choices=['kfactor', 'inputmatrix', 'serial'])
parser.add_argument('-t', '--consecutive', dest='consecutive', action='store')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
parser.add_argument('-w', '--wref', dest='wref', action='store')
parser.add_argument('-x', '--maxiterations', dest='maxiterations', action='store')
args = parser.parse_args()
GPUPROGRAMPATH = args.gpuprogrampath
JOBDIR = args.jobdir
os.chdir(JOBDIR)
INPUTFILE = args.inputfile
mink = int(args.mink)
maxk = int(args.maxk)
seed_list = range(int(args.startseed), int(args.startseed) + int(args.seeds))
#mpitasks = int(args.mpitasks)
JOBDIR = args.jobdir
debug = args.verbose
# read INPUTFILE as gct file
gct_data = NP_GCT(args.inputfile)
vref = args.vref
href = args.href
wref = args.wref
if args.klerrordiffmax:
  klerrordiffmax = float(args.klerrordiffmax)
else:
  klerrordiffmax = None

V = gct_data.data

M = V.shape[1]
#results = []
print("Read " + args.inputfile + "  " + str(V.shape))

if debug:
  print('M ({}) from ({})'.format(M,V))


k_values = np.arange(mink, maxk +1)
if args.parastrategy == 'kfactor':
  k_subsets = divide_almost_equally(k_values, mpi_size)
  my_k_indices = k_subsets[mpi_rank]
else:
  my_k_indices = k_values
try:
  for k in my_k_indices:
    if debug:
      print('start of loop for k={}'.format(k))
    print(f'{mpi_rank}: start of loop for k={k}\n')
    together_counts = numpy.zeros((M,M))
    for seed in seed_list:
      print(f'{mpi_rank}: doing k={k}, seed={seed}\n')
      if debug:
        DEBUGOPTION = '--verbose'
        DEBUGVAL = True
      else:
        DEBUGOPTION = ''
        DEBUGVAL = False
      
      start = time.process_time() 
      WH = runnmf(inputmatrix=V, kfactor=k, checkinterval=int(args.interval), threshold=int(args.consecutive), maxiterations=int(args.maxiterations), seed=seed, debug=DEBUGVAL, comm=comm, parastrategy=args.parastrategy, klerrordiffmax=klerrordiffmax)
      
      # print result and write files only if mpi_rank == 0, or parastrategy
      # is serial or kfactor
      if mpi_rank == 0 or args.parastrategy in ('serial', 'kfactor'):
        #print("xxxxxxxxxxxxxxx Elapsed time for k=" + str(k) + ' seed=' + str(seed) + ": " + str(time.process_time() - start));
        print(f'{mpi_rank}: xxxxxxxxxxxxxxx Elapsed time for k={k} seed={seed} : {time.process_time() - start}\n');
      if debug:
        print("return from runnmf is " + str(WH)) 
      if (not WH):
          #sys.exit(1)
          print(f'failed to get WH ({WH}), continuing...\n')
          continue
      i = cp.asnumpy(cp.argmax(WH[1], axis=0))
      together_counts[i[:, None] == i[None, :]] += 1

    if not args.noconsensus:
      if mpi_rank == 0 or args.parastrategy in ('serial', 'kfactor'):
        print(f'{mpi_rank}: finished all seed trials for k={k}, calculating cophenetic correlation distance...\n')
        # for MPI scatter/gather
        #results.append(together_counts)
        numpy.set_printoptions(threshold=M*M)
        print('consensus matrix shape ({})'.format(together_counts.shape))
        #print(f'together_counts: ({together_counts})\n')
        sys.stdout.write('consensus matrix:')
        for i_index in range(M):
          sys.stdout.write('\n')
          for j_index in range(M):
            sys.stdout.write('{:>2.0f}'.format(together_counts[i_index, j_index]/10))
        sys.stdout.write('\n')
      
      
        i_counts = together_counts.astype(int)    
        consensus_gct = NP_GCT(data=i_counts, rowNames=gct_data.columnnames, colNames=gct_data.columnnames)
        consensus_gct.write_gct('{}.consensus.k.{}.gct'.format(args.outputfileprefix,k))
    
        linkage_mat = scipy.cluster.hierarchy.linkage(together_counts)
        cdm = scipy.spatial.distance.pdist(together_counts)
        cophenetic_correlation_distance, cophenetic_distance_matrix = scipy.cluster.hierarchy.cophenet(linkage_mat, cdm)
        print('k={}, cophenetic_correlation_distance: ({})'.format(k,cophenetic_correlation_distance))
        
        # sort the samples in the consensus matrix for the plot
        countsdf=pd.DataFrame(i_counts, columns=gct_data.columnnames, index=gct_data.columnnames)
        kmeans = cluster.KMeans(n_clusters=2).fit(countsdf)
        labels = kmeans.labels_
    
        namedf = pd.DataFrame(labels, index = gct_data.columnnames)
        sortedNames = namedf.sort_values(0).index
    
        countsdf = countsdf[sortedNames]
        countsdf = countsdf.reindex(sortedNames)
        sorted_i_counts = countsdf.to_numpy()
        sc = NP_GCT(data=sorted_i_counts, rowNames=sortedNames, colNames=sortedNames )
        sc.write_gct('{}.consensus.k.{}.sorted.gct'.format(args.outputfileprefix,k))
    
    
        fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        im = plt.imshow(sorted_i_counts, cmap='bwr', interpolation='nearest')
    
        ax.set_xticks(np.arange(len(sortedNames)), labels=sortedNames)
        ax.set_yticks(np.arange(len(sortedNames)), labels=sortedNames)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  rotation_mode="anchor")
    
        ax.set_title("Consensus Matrix, k="+str(k))
        fig.tight_layout()
    
        plt.savefig('{}.consensus.k.{}.pdf'.format(args.outputfileprefix,k))  
        with open('{}.cophentic.txt'.format(args.outputfileprefix), 'w') as file:
            file.write(str(k) + "\t" + str(cophenetic_correlation_distance) + "\n")

except:
  traceback.print_tb(sys.exc_info()[2])
  print("Unexpected error:", sys.exc_info()[0])
  if args.nocleanup == True:
    print('keeping ' + JOBDIR + '/k*')
    print('keeping ' + JOBDIR + '/bionmf.input.txt')
  #else:
    #if not debug:
      #os.unlink(JOBDIR + '/bionmf.input.txt')
      #for kdir in kdirs:
      #  print('rmtree of ' + JOBDIR + '/k.{}'.format(k))
      #  shutil.rmtree(JOBDIR + '/k.{}'.format(k))
  raise

if args.keepintermediatefiles == True:
  print('keeping ' + JOBDIR + '/bionmf.input.txt')
else:
  print('unlink of ' + JOBDIR + '/bionmf.input.txt')
  os.unlink(JOBDIR + '/bionmf.input.txt')
