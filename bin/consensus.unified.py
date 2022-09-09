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
# https://docs.cupy.dev/en/stable/user_guide/performance.html
# Pandas on GPU:
# https://developer.nvidia.com/blog/scikit-learn-tutorial-beginners-guide-to-gpu-accelerating-ml-pipelines/
# cuDF requires CUDA >=11.0, does not have pip install:
# https://github.com/rapidsai/cudf/tree/main#development-setup
# https://github.com/rapidsai/raft
# https://jocelyn-ong.github.io/hierarchical-clustering-in-SciPy/

# import modules
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
import math
import cupyx
import pickle
import pathlib
import h5py
from fastdist import fastdist
import fastcluster

# set data types
RANDTYPE = cp.float32
OTHERTYPE = cp.float32
# seed count more than 255 won't work...
TOGETHERTYPE = cp.uint8
NUMPYTYPE = np.float32
EPSILON = cp.finfo(OTHERTYPE).eps
W = None
H = None

# https://docs.cupy.dev/en/stable/user_guide/kernel.html
# kerneldiv = cp.ElementwiseKernel(
#   'float32 x, float32 y', 'float32 z', 'z = x / y', 'kerneldiv')

# Class for reading and writing GCT files
class NP_GCT:
  # #1.2
  # ros cols
  # name descrip sample1 sample2 ...
  # rowname1 rowdescrip1 value value ...   
  def __init__(self, filename=None, data=None, rowNames=None, rowDescrip=None, colNames=None):
    if filename:
      print(filename)

      # init from passed in stuff
      f = open(filename,'r')
      count=0
      f.readline() ##1.2
      dims = f.readline().split('\t') # rows cols
      if debug:
        print(f'(rows, cols) {dims}\n')
      #data = np.genfromtxt(fname='testfile', dtype=np.half, delimiter="\t", filling_values=0, comments='#####')  # change filling_valuesas req'd to fill in missing values
      #data = np.loadtxt(fname='testfile', dtype=np.half, delimiter="\t")  # change filling_valuesas req'd to fill in missing values
      numCols = int(dims[1])
      colNames = f.readline().strip().split('\t');
      # override comments because there are none in a gct but there can be # in a gene description
      #data = cp.genfromtxt(fname=filename, delimiter="\t", skip_header=3, usecols=range(2,numCols+2), filling_values=0, comments='#####')  # change filling_valuesas req'd to fill in missing values
      #data = cp.loadtxt(fname=filename, delimiter="\t", skiprows=3, usecols=range(2,numCols+2), comments='#####')  # change filling_valuesas req'd to fill in missing values
      #data = np.genfromtxt(fname=filename, delimiter="\t", skip_header=3, usecols=range(2,numCols+2), filling_values=0, comments='#####')  # change filling_valuesas req'd to fill in missing values
      #data = np.genfromtxt(fname=filename, dtype=np.half, delimiter="\t", skip_header=3, usecols=range(2,numCols+2), filling_values=0, comments='#####')  # change filling_valuesas req'd to fill in missing values
      #self.data = data


      self.columnnames = colNames[2:];
      self.rownames = [None] * int(dims[0])
      self.rowdescriptions = [None] * int(dims[0])

      data = np.empty((int(dims[0]), int(dims[1])), dtype=OTHERTYPE)
      if debug:
         print(f'before reading lines from input file\n')
      while True:
          # Get next line from file
          line = f.readline()

          # if line is empty
          # end of file is reached
          if not line:
              break

          line = line.split('\t');
          name = line[0]
          description=line[1]
          for linetuple in enumerate(line[2:]):
              data[count, int(linetuple[0])] = OTHERTYPE(linetuple[1])
          self.rownames[count] = name
          self.rowdescriptions[count] = description
          count += 1
      self.data = data
      #self.data = cp.array(data, dtype=OTHERTYPE)

      f.close()
    else:    
      self.data=data
      self.rownames=rowNames
      self.rowdescriptions=rowDescrip
      self.columnnames=colNames
    print(f'{rank}: Loaded matrix of shape {self.data.shape}\n')
    
  def write_gct(self, file_path):
    """
    Writes the provided NP_GCT to a GCT file.
    If any of rownames, rowdescriptions or columnnames is missing write the 
    index in their place (starting with 1)
    
    :param file_path:
    :return:
    """
    np.set_printoptions(suppress=True)
    with open(file_path, 'w') as file:
      nRows = self.data.shape[0]
      nCols = self.data.shape[1]
      rowNames = self.rownames;
      rowDescriptions = self.rowdescriptions;
      colNames = self.columnnames;
      #if not rowNames:
      if len(rowNames) == 0:
        rowNames = ["{:}".format(n) for n in range(1,self.data.shape[0]+1)]
      if not rowDescriptions:
        rowDescriptions = rowNames
      #if not colNames:
      if len(colNames) == 0:
        colNames =  ["{:}".format(n) for n in range(1,self.data.shape[1]+1)]
      file.write('#1.2\n' + str(nRows) + '\t' + str(nCols) + '\n')
      file.write("Name\tDescription\t")
      file.write(colNames[0])
      for j in range(1, nCols):
        file.write('\t')
        file.write(colNames[j])
      file.write('\n')
      for i in range(nRows):
        file.write(rowNames[i] + '\t')
        file.write(rowDescriptions[i] + '\t')
        file.write(str(self.data[i,0]))
        for j in range(1, nCols):
          file.write('\t')
          file.write(str(self.data[i,j]))
        file.write('\n')
    print("File written " + file_path)
  def __write_gct(self, file_path):
    """
    Writes the provided DataFrame to a GCT file.
    Assumes that the DataFrame matches the structure of those produced
    by the GCT() function in this library
    :param df:
    :param file_path:
    :return:
    """
    np.set_printoptions(suppress=True)
    with open(file_path, 'w') as file:
            
      file.write('#1.2\n' + str(len(self.rownames)) + '\t' + str(len(self.columnnames)) + '\n')
      file.write("Name\tDescription\t")
      file.write(self.columnnames[0])
            
      for j in range(1, len(self.columnnames)):
        file.write('\t')
        file.write(self.columnnames[j])
                
      file.write('\n')
                
      for i in range(len(self.rownames)):
        file.write(self.rownames[i] + '\t')
        file.write(self.rowdescriptions[i] + '\t')
        file.write(str(self.data[i,0]))
        for j in range(1, len(self.columnnames)):
          file.write('\t')
          file.write(str(self.data[i,j]))
                    
        file.write('\n')
    print("File written " + file_path)

# write numpy format arrays, with associated attributes pickle
def write_npy(data=None, outputfileprefix=None, colNames=[], rowNames=[], rowDescrip = [], datashape = None):
  attribute_dict = { 'column_names' : colNames,
                     'row_names' : rowNames,
                     'row_descriptions' : rowDescrip,
                     'data_shape' : datashape,
                   }
  FO = open(outputfileprefix + '.npy.attributes', 'wb')
  pickle.dump(attribute_dict, FO, protocol=0)
  FO.close()
  FO = open(outputfileprefix + '.npy', 'wb')
  numpy.save(FO, data)
  FO.close()
         
# write HDF5 array, including attributes
def write_h5(data=None, outputfileprefix=None, colNames=[], rowNames=[], rowDescrip = [], datashape = None, comm_world=None):
  rank = comm_world.rank
  print(f'{rank}: beginning of write_h5\n')
  outf = h5py.File(outputfileprefix + '.h5', 'w', driver='mpio', comm=comm_world, libver='latest')
  print(f'{rank}: after h5py.File\n')
  datasetname = outputfileprefix
  print(f'{rank}: before create_dataset({datasetname})\n')
  dset = outf.create_dataset(datasetname, data=data, dtype=data.dtype)
  print(f'{rank}: after create_dataset({datasetname})\n')
  print(f'colNames {colNames}\n')
  dset.attrs['column_names'] = colNames
  dset.attrs['row_names'] = rowNames
  dset.attrs['row_descriptions'] = rowDescrip
  dset.attrs['data_shape'] = datashape
  outf.close()
         
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

# For a given kfactor and seed, return W and H
def runnmf(myVcols=None,myVrows=None, mystartrow=None, myendrow=None,mystartcol=None, myendcol=None, Wsendcountlist=None, Hsendcountlist=None,kfactor=2,checkinterval=10,threshold=40,maxiterations=2000,seed=1,debug=False, comm=None, parastrategy='serial', klerrordiffmax=None):
  thistime = MPI.Wtime()
  #print(f'rank {rank}: finished all k loops: ({thistime - lasttime})\n')
  lasttime = thistime
  global W
  global H
  W = None
  H = None
  mempool = cp.get_default_memory_pool()
  mempool.free_all_blocks()
  # disable memory pool:
  #cp.cuda.set_allocator(None)
  #cp.cuda.runtime.setDevice(rank)
  if debug:
    print(f'{rank}: cp.cuda.runtime.getDevice() ({cp.cuda.runtime.getDevice()})\n')
  # seed the PRNG
  cp.random.seed(seed)
  if debug:
    thistime = MPI.Wtime()
    print(f'rank {rank}: setup time: ({thistime - lasttime})\n')
    lasttime = thistime
  # create H and W random on device
  # H = M (inputarray.shape[1]) x k (stored transposed), W = N (inputarra.shape[0]x k
  # in the bionmf-gpu code,
  # set_random_values( p_dH, nrows, K, Kp,
  # set_random_values( d_W, N, K, Kp,
  # should be transposed for H...
  # Is there any danger of H and W generated from rand()
  # getting out of sync amongst the tasks?
  # this is over interval [0, 1), might need to add a smidge...
  H = cp.array(cp.random.rand(kfactor,M, dtype=RANDTYPE), dtype=OTHERTYPE) + EPSILON
  # is it better to create the transposed matrix from the start?
  Ht = H.transpose()
  W = cp.array(cp.random.rand(N,kfactor,dtype=RANDTYPE), dtype=OTHERTYPE) + EPSILON
  Wt = W.transpose()
  if debug:
    print(f'{rank}: initial H: ({H})\n')
    print(f'{rank}: initial Ht: ({Ht})\n')
    print(f'{rank}: initial W: ({W})\n')
    print(f'{rank}: initial Wt: ({Wt})\n')
    print(f'{rank}: V: ({V})\n')
    print(f'{rank}: WH: ({cp.matmul(W,H)})\n')
    # write out pickled W and H
    cp.asnumpy(W).dump(f'k-{kfactor}.seed-{seed}.initialW.pkl')
    cp.asnumpy(H).dump(f'k-{kfactor}.seed-{seed}.initialH.pkl')
    thistime = MPI.Wtime()
    print(f'rank {rank}: random W and H creation time: ({thistime - lasttime})\n')
    lasttime = thistime
  iterationcount = 0
  oldclassification = None
  sameclassificationcount = 0
  oldserialerror = None
  oldmpierror = None
  #KLFO = open('kldiv.tsv', 'w')
  if debug:
    print(f'{rank}:{iterationcount} before iterating, mempool.used_bytes(): {mempool.used_bytes()}\n')
    print(f'{rank}:{iterationcount} before iterating, mempool.total_bytes(): {mempool.total_bytes()}\n')
  while iterationcount < maxiterations:
    if debug:
      print(f'{rank}:{iterationcount} start of iteration, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}:{iterationcount} start of iteration, mempool.total_bytes(): {mempool.total_bytes()}\n')
      thistime = MPI.Wtime()
      print(f'rank {rank}: random W and H creation time: ({thistime - lasttime})\n')
      lasttime = thistime
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
      print(f'Whmm.dtype: ({WHm.dtype})\n')
      print(f'{rank}: after matmul, WHm.shape: {WHm.shape}\n')
      print(f'{rank}: WHm: ({WHm})\n')
      print(f'{rank}: update H, matmul(W, H[:,mystartcol:myendcol + 1]), WHm: ({WHm})\n')
    # AUX = V (input matrix) ./ (W*H)
    # AUX (N x M)
    # no nditer in cupy
    # https://numpy.org/doc/stable/reference/arrays.nditer.html
    # https://stackoverflow.com/questions/42190783/what-does-three-dots-in-python-mean-when-indexing-what-looks-like-a-number
    #with cp.nditer(myVcols, flags=['multi_index'], op_flags=['readwrite']) as it:
    WHm = cp.divide(myVcols, WHm)
    mempool.free_all_blocks()
    if debug:
      print(f'{rank}: after divide, WHm.shape: {WHm.shape}\n')
      print(f'{rank}: update H, divide(V[:,mystartcol:myendcol + 1], WHm), WHm: ({WHm})\n')
      print(f'{rank}: WHm.dtype ({WHm.dtype})\n')
      print(f'{rank}: Wt.dtype ({Wt.dtype})\n')
    # WTAux = Wt * AUX
    if debug:
      print(f'{rank}: Wt: ({Wt})\n')
    WTAUX = cp.matmul(Wt, WHm)
    if debug:
      print(f'{rank}: Wt.shape {Wt.shape} WHm.shape {WHm.shape} WTAUX.shape {WTAUX.shape}\n')
      print(f'{rank}:  WTAUX.shape: {WTAUX.shape}\n')
      print(f'{rank}: WTAUX: ({WTAUX})\n')
      print(f'{rank}: update H, matmul(Wt, WH), WTAUX: ({WTAUX})\n')
    # how do we get reduced an accumulated ACCWT below?
    # sum each column down to a single value...
    ACCW = cp.sum(W, axis=0)
    # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
    # WTAUXDIV = WTAUX ./ ACCWT
    if debug:
      print(f'{rank}:  WTAUX.shape: {WTAUX.shape}, ACCW.shape: {ACCW.shape}\n')
      print(f'{rank}: ACCW: ({ACCW})\n')
    WTAUXDIV = cp.divide(WTAUX.transpose(), ACCW)
    WTAUXDIV = WTAUXDIV.transpose()
    if debug:
      print(f'{rank}: WTAUXDIV: ({WTAUXDIV})\n')
    # H = H .* WTAUXDIV
    Hnewnan = cp.multiply(H[:,mystartcol:myendcol + 1], WTAUXDIV)
    Hnew = cp.nan_to_num(Hnewnan, copy=False, nan=EPSILON)
    Hnewnan = None
    WTAUX = None
    WTAUXDIV = None
    ACCW = None
    mempool.free_all_blocks()
    if debug:
      print(f'{rank}:{iterationcount} after update Hnew, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}:{iterationcount} after update Hnew, mempool.total_bytes(): {mempool.total_bytes()}\n')
    # sync H to all devices
    if debug:
      thistime = MPI.Wtime()
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} : update Hnew time: ({thistime - lasttime})\n')
      lasttime = thistime
    if args.parastrategy == 'inputmatrix':
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
      Hrecv = cp.empty(H.size, dtype=OTHERTYPE)
      if debug:
        print(f'{rank}: Hnew {Hnew}\n')
        print(f'{rank}: Hnewflat {Hnewflat}\n')
        print(f'{rank}: H.shape {H.shape}\n')
        print(f'{rank}: Hrecv.shape {Hrecv.shape}\n')
        print(f'{rank}: Hsendcountlist {Hsendcountlist}\n')
      cp.cuda.Stream.null.synchronize()
      comm.Allgatherv(sendbuf=Hnewflat,recvbuf=(Hrecv,Hsendcountlist))
      if debug:
        print(f'{rank}: after Allgatherv, Hrecv.shape {Hrecv.shape}\n')
        print(f'{rank}: after Allgatherv, Hrecv {Hrecv}\n')
      Hnewfull = Hrecv.reshape(Hshape, order='F')
      Hnewflat = None
      Hrecv = None
      mempool.free_all_blocks()
      if debug:
        print(f'{rank}: after Allgatherv, reshape, H.shape {H.shape}\n')
        print(f'{rank}: after Allgatherv, reshape, H {H}\n')
      H = Hnewfull
      Hnewfull = None
      mempool.free_all_blocks()
    else:
      cp.cuda.Stream.null.synchronize()
      H = Hnew
    if debug:
      thistime = MPI.Wtime()
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} : sync H time: ({thistime - lasttime})\n')
      lasttime = thistime
    Ht = H.transpose()
    if debug:
      print(f'{rank}:{iterationcount} after sync H, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}:{iterationcount} after sync H, mempool.total_bytes(): {mempool.total_bytes()}\n')
    if debug:
      print(f'{rank}: before update W\n')
    # update W
    # * WH(BLN,Mp) = W(BLN,Kp) * H
    # * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
    # * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    # generate ACCUMH
    ACCH = cp.sum(H, axis=1, dtype=OTHERTYPE)
    if debug:
      print(f'{rank}: H: ({H})\n')
      print(f'{rank}: ACCH: ({ACCH})\n')
    WHm = cp.matmul(W[mystartrow:myendrow + 1,:], H)
    WHm = cp.divide(myVrows, WHm)
    # from update_W notes:
    #  * Waux(BLN,Kp) = WH(BLN,Mp) * H'
    # should I be using the original Ht here?
    HTAUX = cp.matmul(WHm, Ht)
    if debug:
      print(f'{rank}: HTAUX: ({HTAUX})\n')
    # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
    WWAUXnan = cp.multiply(W[mystartrow:myendrow + 1,:], HTAUX)
    WWAUX = cp.nan_to_num(WWAUXnan, copy=False, nan=EPSILON)
    WWAUXnan = None
    HTAUX = None
    WHm = None
    if debug:
      print(f'{rank}: WWAUX: ({WWAUX})\n')
    Wnewnan = cp.divide(WWAUX, ACCH)
    Wnew = cp.nan_to_num(Wnewnan, copy=False, nan=EPSILON)
    Wnewnan = None
    WWAUX = None
    ACCH = None
    mempool.free_all_blocks()
    if debug:
      print(f'{rank}: Wnew: ({Wnew})\n')
      print(f'{rank}: Hnew: ({Hnew}), Hnew.shape: {Hnew.shape}\n')
      print(f'{rank}: Wnew: ({Wnew}), Wnew.shape: {Wnew.shape}\n')
      print(f'{rank}: Wnew: ({Wnew})\n')
      print(f'{rank}:{iterationcount} after update Wnew, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}:{iterationcount} after update Wnew, mempool.total_bytes(): {mempool.total_bytes()}\n')
    # sync W to all devices
    if debug:
      thistime = MPI.Wtime()
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} :  update Wnew time: ({thistime - lasttime})\n')
      lasttime = thistime
    if args.parastrategy == 'inputmatrix':
      Wshape = W.shape
      Wnewflat = Wnew.ravel(order='C')
      Wrecv = cp.empty(W.size, dtype=OTHERTYPE)
      if debug:
        print(f'{rank}: Wnew {Wnew}\n')
        print(f'{rank}: Wnewflat {Wnewflat}\n')
        print(f'{rank}: W.shape {W.shape}\n')
        print(f'{rank}: Wrecv.shape {Wrecv.shape}\n')
      cp.cuda.Stream.null.synchronize()
      comm.Allgatherv(sendbuf=Wnewflat,recvbuf=(Wrecv, Wsendcountlist))
      Wnewfull = Wrecv.reshape(Wshape, order='C')
      Wnewflat = None
      Wrecv = None
      W = Wnewfull
      Wnewfull = None
      mempool.free_all_blocks()
    else:
      cp.cuda.Stream.null.synchronize()
      H = Hnew
      W = Wnew
    if debug:
      print(f'{rank}:{iterationcount} after sync W, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}:{iterationcount} after sync W, mempool.total_bytes(): {mempool.total_bytes()}\n')
    if debug:
      print(f'{rank}: after Allgatherv, W.shape {W.shape}\n')
      print(f'{rank}: after Allgatherv, W {W}\n')
      thistime = MPI.Wtime()
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} : sync W time: ({thistime - lasttime})\n')
      lasttime = thistime
    Wt = W.transpose()
    if debug:
      thistime = MPI.Wtime()
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} : W transpose time: ({thistime - lasttime})\n')
      lasttime = thistime
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
      classstart = time.process_time() 
      # check KL divergence
      if debug:
        print(f'{rank}:{iterationcount} start classification check, mempool.used_bytes(): {mempool.used_bytes()}\n')
        print(f'{rank}:{iterationcount} start classification check, mempool.total_bytes(): {mempool.total_bytes()}\n')
      if not klerrordiffmax == None:
        if debug:
          print(f'{rank}: checking KL divergence, since klerrordiffmax: ({klerrordiffmax})\n')
        if args.parastrategy == 'inputmatrix':
          if debug:
            print(f'{rank}: checking MPI KL divergence, since parastrategy: ({parastrategy})\n')
          WHkl = cp.dot(W,Hnew)
          if debug:
            print(f'{rank}: WHkl.shape ({WHkl.shape})\n')
          WH_datakl = WHkl.ravel()
          if debug:
            print(f'{rank}: myVcols.shape {myVcols.shape}\n')
          X_datakl = myVcols.ravel()
          if debug:
            print(f'{rank}: WH_datakl: ({WH_datakl})\n')
            print(f'{rank}: X_datakl: ({X_datakl})\n')
            print(f'EPSILON ({EPSILON})\n')
          # instead of indices, should set to EPSILON, like WH_datalk...
          indices = X_datakl > EPSILON
          #antiindices = X_datakl <= EPSILON
          #if debug:
          #  print(f'{rank}: antiindices: ({antiindices})\n')
          # assume everything set to EPSILON...
          WH_datakl[WH_datakl == 0] = EPSILON
          if debug:
            print(f'{rank}: after indices and EPSILON, WH_datakl: ({WH_datakl})\n')
            print(f'{rank}: after indices and EPSILON, X_datakl: ({X_datakl})\n')
            print(f'X_datakl ({X_datakl})\n')
          sum_WH = cp.dot(cp.sum(W, axis=0), cp.sum(Hnew, axis=1))
          div = WH_datakl
          div = cp.divide(X_datakl,WH_datakl)
          #if debug:
          #  print(f'{rank}: X_datakl / WH_datakl, div: ({div})\n')
          cp.log(div,out=div)
          res = None
          mempool.free_all_blocks()
          res = cp.dot(X_datakl, div, out=res)
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
          if debug:
            print(f'{rank}:{iterationcount} plain error {error}\n')
          error = numpy.sqrt(2 * totalres/X_datakl.size) * math.sqrt(X_datakl.size)
          if debug:
            print(f'{rank}:{iterationcount} scaled error {error}\n')
          if type(oldmpierror) == type(None):
            errordiff = None
            oldmpierror = error
          else:
            errordiff = oldmpierror - error
            if debug:
              print(f'{rank}:{iterationcount} oldmpierror {oldmpierror} error {error} errordiff {errordiff}\n')
            oldmpierror = error
          if debug:
            print(f'{rank}: afer Reduce, totalres: ({totalres})\n')
            print(f'{rank}: MPI KL divergence, error: ({error})\n')
            print(f'{rank}: mpi error difference: {errordiff}\n')
          if (not type(errordiff) == type(None)) and errordiff < klerrordiffmax:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) < klerrordiffmax ({klerrordiffmax}), return(W,H)\n')
            totalres = None
            res = None
            indices = None
            sum_WH = None
            WHkl = None
            WH_datakl = None
            X_datakl = None
            div = None
            if errordiff >= 0.0:
              return(W,H)
            else:
              if debug:
                print(f'KL error is increasing before reaching klerrordiffmax ({klerrordiffmax}), not returning WH!')
          else:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) not less than klerrordiffmax ({klerrordiffmax})\n')
            totalres = None
            res = None
            indices = None
            sum_WH = None
            WHkl = None
            WH_datakl = None
            X_datakl = None
            div = None
        #end else if parastrategy == 'inputmatrix':
        else:
          if debug:
            print(f'{rank}: checking serial KL divergence, since parastrategy: ({parastrategy})\n')
          cp.cuda.Stream.null.synchronize()
          WHkl = cp.dot(W,H)
          cp.cuda.Stream.null.synchronize()
          WH_datakl = WHkl.ravel()
          cp.cuda.Stream.null.synchronize()
          #X_datakl = V.ravel()
          X_datakl = myVcols.ravel()
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
          res += sum_WH - (X_datakl / X_datakl.size).sum() * X_datakl.size
          cp.cuda.Stream.null.synchronize()
          if debug:
            print(f'{rank}: KL divergence, serial calculation, res: ({res})\n')
            print(f'{rank}: typ(res): ({type(res)})\n')
            print(f'{rank}: type(2 * res): ({type(2 * res)})\n')
            print(f'{rank}: type(cp.sqrt(2 * res)) {type(cp.sqrt(2 * res))}')
          #error = OTHERTYPE(cp.sqrt(2 * res))
          #error = cp.sqrt(2 * res)
          totalres = res
          error = numpy.sqrt(2 * totalres)
          #error = numpy.sqrt(2 * totalres/X_datakl.size) * math.sqrt(X_datakl.size)
          if debug:
            print(f'{rank}: oldserialerror: {oldserialerror}\n')
          if type(oldserialerror) == type(None):
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
          if (not type(errordiff) == type(None)) and errordiff < klerrordiffmax:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) < klerrordiffmax ({klerrordiffmax}), return(W,H)\n')
            if errordiff >= 0.0:
              WHkl = None
              WH_datakl = None
              X_datakl = None
              div = None
              res = None
              indices = None
              sum_WH = None
              mempool.free_all_blocks()
              return(W,H)
            else:
              print(f'KL error is increasing before reaching klerrordiffmax ({klerrordiffmax}), no returning WH!')
          else:
            if debug:
              print(f'{rank}: interationcount ({iterationcount}): errordiff ({errordiff}) not less than klerrordiffmax ({klerrordiffmax})\n')
            WHkl = None
            WH_datakl = None
            X_datakl = None
            div = None
            res = None
            indices = None
            sum_WH = None
            mempool.free_all_blocks()
            #return(W,H)
        #end else if parastrategy != 'inputmatrix':
        WHkl = None
        WH_datakl = None
        X_datakl = None
        div = None
        mempool.free_all_blocks()
      # else not klerrordiffmax == None:
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
              print(f'{rank}:{iterationcount} classification unchanged in {sameclassificationcount} trials,breaking.\n')
              print(f'rank {rank}:kfactor {kfactor}:seed {seed}: iteration{iterationcount} classification unchanged in {sameclassificationcount} trials,breaking.\n')
            #cp.savetxt(os.path.basename(args.inputmatrix) + '_H.txt', H)
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
      if debug:
        print(f'{rank}:{iterationcount} end classification check, mempool.used_bytes(): {mempool.used_bytes()}\n')
        print(f'{rank}:{iterationcount} end start classification check, mempool.total_bytes(): {mempool.total_bytes()}\n')
    thistime = MPI.Wtime()
    if debug:
      print(f'rank {rank}: kfactor {kfactor}: seed {seed} : iteration {iterationcount} : divergence/classification check time: ({thistime - lasttime})\n')
    lasttime = thistime
    iterationcount = iterationcount + 1
  #KLFO.close()
  if debug:
    print(f'{rank}: iterationcount ({iterationcount})\n')

# start of execution

# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.rank
numtasks = comm.size

# on shared GPU nodes, assume the job sees only its device ids,
# starting from 0, incrementing by one.
#print(f'{rank}: cp.cuda.runtime.getDeviceCount() {cp.cuda.runtime.getDeviceCount()}\n')
#print(f'{rank}: before setDevice, cp.cuda.runtime.getDevice() {cp.cuda.runtime.getDevice()}\n')
for deviceid in range(cp.cuda.runtime.getDeviceCount()):
  if rank == deviceid:
    cp.cuda.runtime.setDevice(deviceid)
#print(f'{rank}: after setDevice, cp.cuda.runtime.getDevice() {cp.cuda.runtime.getDevice()}\n')
# https://docs.cupy.dev/en/stable/user_guide/memory.html
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
lasttime = MPI.Wtime()

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
parser.add_argument('-q', '--inputfiletype', dest='inputfiletype', action='store')
parser.add_argument('-r', '--seeds', dest='seeds', action='store')
parser.add_argument('-s', '--parastrategy', dest='parastrategy', action='store', choices=['kfactor', 'inputmatrix', 'serial'])
parser.add_argument('-t', '--consecutive', dest='consecutive', action='store')
parser.add_argument('-u', '--outputfiletype', dest='outputfiletype', action='store')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
parser.add_argument('-x', '--maxiterations', dest='maxiterations', action='store')
#parser.add_argument('-z', '--postprocess', dest='postprocess', action='store_true')
args = parser.parse_args()
JOBDIR = args.jobdir
os.makedirs(JOBDIR, exist_ok=True)
os.chdir(JOBDIR)
INPUTFILE = args.inputfile
mink = int(args.mink)
maxk = int(args.maxk)
seed_list = range(int(args.startseed), int(args.startseed) + int(args.seeds))
JOBDIR = args.jobdir
debug = args.verbose
if debug:
  print(f'mempool.used_bytes(): ({mempool.used_bytes()})\n')
  print(f'mempool.total_bytes(): ({mempool.total_bytes()})\n')
  print(f'mempool.n_free_blocks(): ({mempool.n_free_blocks()})\n')
if args.inputfiletype == 'h5':
  print(f'Sorry, HDF5 format not supported, yet!\n')
  sys.exit(1)
if args.outputfiletype == 'h5':
  print(f'Sorry, HDF5 format not supported, yet!\n')
  sys.exit(1)
if debug:
  print(f'{rank}: before reading input, mempool.used_bytes(): {mempool.used_bytes()}\n')
  print(f'{rank}: before reading input, mempool.total_bytes(): {mempool.total_bytes()}\n')
if args.inputfiletype in ('gct',):
  # read INPUTFILE as gct file
  if debug:
    print(f'{rank}: reading input file with NP_GCT({args.inputfile})\n')
  gct_data = NP_GCT(args.inputfile)
  if debug:
    print(f'{rank}: done reading input file with NP_GCT({args.inputfile})\n')
  V = gct_data.data
elif args.inputfiletype in ('h5',):
  print(f'HDF5 input format not supported, yet!\n')
  sys.exit(1)
elif args.inputfiletype in ('npy',):
  # read INPUTFILE as npy file
  print(f'{rank}: reading input file with np.array(np.load({args.inputfile})\n')
  V = np.array(np.load(args.inputfile), dtype=OTHERTYPE)
  # if 'npy', look for inputfile.attributes
  if args.inputfiletype in ('npy',):
    inputattributespath = pathlib.Path(args.inputfile + '.attributes')
    if inputattributespath.exists:
      print(f'{rank}: found attributes file ({args.inputfile + ".attributes"})\n')
      FO = open(args.inputfile + '.attributes','rb')
      attributes_dict = pickle.load(FO)
      FO.close()
    else:
      attributes_dict = None
  if debug:
    print(f'{rank}: done reading input file with np.array(np.load({args.inputfile})\n')

# for input matrix, set minimum EPSILON for all elements
V[V < EPSILON] = EPSILON

if args.klerrordiffmax:
  klerrordiffmax = NUMPYTYPE(args.klerrordiffmax)
else:
  klerrordiffmax = None

M = V.shape[1]
#print("Read " + args.inputfile + "  " + str(V.shape))

if debug:
  print('M ({}) from ({})'.format(M,V))

# depending on whether we are decomposing the work by whole kfactor
# to each task or decomposing the input matrix across all tasks,
# create myVcols, mVrows whole/partial inputs
k_values = np.arange(mink, maxk + 1)
N = V.shape[0]
M = V.shape[1]
if debug:
  DEBUGOPTION = '--verbose'
  DEBUGVAL = True
else:
  DEBUGOPTION = ''
  DEBUGVAL = False
if args.parastrategy == 'kfactor':
  k_subsets = divide_almost_equally(k_values, numtasks)
  my_k_indices = k_subsets[rank]
  # need to create myVcols and myVrows here...
  myV = cp.array(V)
  myVcols = myV
  myVrows = myV
  mystartcol = 0
  myendcol = M - 1
  mystartrow = 0
  myendrow = N - 1
else:
  my_k_indices = k_values
  # here is where to put decomposition according to parastrategy...
  if comm == None or args.parastrategy in ('serial', 'kfactor'):
    rank = 0
    numtasks = 1
  else:
    rank = comm.Get_rank()
    numtasks = comm.Get_size()
  if debug:
    print(f'{rank}: inputarray.shape: ({V.shape})\n')
    print(f'{rank}: V: ({V})\n')
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
  myrowcount = (myendrow + 1) - mystartrow
  mycolcount = (myendcol + 1) - mystartcol
  # do I need to add the padded rows and columns to the last task?
  # or are the matrix operations okay?

try:
  if debug:
    thistime = MPI.Wtime()
    print(f'from beginning to start of for k loop: ({thistime - lasttime})\n')
    lasttime = thistime
    print(f'{rank}: before iterating over k, mempool.used_bytes(): {mempool.used_bytes()}\n')
    print(f'{rank}: before iterating over k, mempool.total_bytes(): {mempool.total_bytes()}\n')
  
  # iterate over kfactors
  kdirs = []
  for k in my_k_indices:
    kdirs.append('k.{}'.format(k))
    os.chdir(JOBDIR)
    if args.keepintermediatefiles == True:
      os.makedirs(f'k.{k}', exist_ok=True)
    kstart = time.process_time() 
    if debug:
      print(f'{rank}: k {k} start of k iteration, mempool.used_bytes(): {mempool.used_bytes()}\n')
      print(f'{rank}: k {k} start of k iteration, mempool.total_bytes(): {mempool.total_bytes()}\n')
    myVcols = cp.array(V[:,mystartcol:myendcol + 1])
    myVrows = cp.array(V[mystartrow:myendrow + 1,:])
    if debug:
      print(f'{rank}: start of loop for k={k}\n')
    
    if args.parastrategy == 'inputmatrix':
      # set up Hsendcountlist and Wsendcountlist for use in AllGatherv
      Hsendcountlist = []
      for tn in range(numtasks):
        if colremainder == 0:
          Hsendcountlist.append(k * colspertask)
        else:
          if tn == numtasks - 1:
            #last task:
            Hsendcountlist.append(k * ((colspertask + 1) - colpad))
          else:
            Hsendcountlist.append(k * (colspertask + 1))
      Wsendcountlist = []
      for tn in range(numtasks):
        if rowremainder == 0:
          Wsendcountlist.append(k * rowspertask)
        else:
          if tn == numtasks - 1:
            #last task:
            Wsendcountlist.append(k * ((rowspertask + 1) - rowpad))
          else:
            Wsendcountlist.append(k * (rowspertask + 1))
    else:
      Hsendcountlist = None
      Wsendcountlist = None
    if debug:
      print(f'{rank}: mystartrow: {mystartrow}, myendrow: {myendrow}, mystartcol: {mystartcol}, myendcol: {myendcol}, myrowcount: {myrowcount}, mycolcount: {mycolcount}, Hsendcountlist: {Hsendcountlist}, Wsendcountlist: {Wsendcountlist}\n')
    # allocate the consensus matrix on device
    together_counts = cp.zeros((M,M),dtype=TOGETHERTYPE)
    # iterate over seeds
    for seed in seed_list:
      start = time.process_time() 
      os.chdir(JOBDIR)
      if args.keepintermediatefiles == True:
        os.makedirs(f'k.{k}/seed.{seed}', exist_ok=True)
        os.chdir(f'k.{k}/seed.{seed}')
      if debug:
        print(f'{rank}: doing k={k}, seed={seed}\n')
        print(f'{rank}: k {k} : seed {seed} start of seed iteration, mempool.used_bytes(): {mempool.used_bytes()}\n')
        print(f'{rank}: k {k} : seed {seed} start of seed iteration, mempool.total_bytes(): {mempool.total_bytes()}\n')
      W,H = runnmf(myVcols=myVcols, myVrows=myVrows, mystartrow=mystartrow, myendrow=myendrow, mystartcol=mystartcol, myendcol=myendcol, Hsendcountlist=Hsendcountlist, Wsendcountlist=Wsendcountlist, kfactor=k, checkinterval=int(args.interval), threshold=int(args.consecutive), maxiterations=int(args.maxiterations), seed=seed, debug=DEBUGVAL, comm=comm, parastrategy=args.parastrategy, klerrordiffmax=klerrordiffmax)
      # print result and write files only if rank == 0, or parastrategy
      # is serial or kfactor
      if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
        if debug:
          print(f'{rank}: xxxxxxxxxxxxxxx Elapsed time for k={k} seed={seed} : {time.process_time() - start}\n');
      if type(H) == type(None):
        print(f'failed to get H ({H}), continuing...\n')
        #sys.exit(1)
        continue
      else:
        #print(f'type(H) not type(None)...')
        if args.keepintermediatefiles == True:
          if args.inputfiletype in ('npy',) and inputattributespath.exists:
            columnnames = attributes_dict['column_names']
            rownames = attributes_dict['row_names']
            rowdescriptions = attributes_dict['row_descriptions']
            Wdatashape = [N,k]
            Hdatashape = [k,M]
          elif args.inputfiletype in ('gct',):
            columnnames = gct_data.columnnames
            rownames = gct_data.rownames
            rowdescriptions = gct_data.rowdescriptions
            Wdatashape = [N,k]
            Hdatashape = [k,M]
          if args.outputfiletype == 'npy':
            write_npy(W.get(), f'{args.outputfileprefix}.W.k.{k}.seed.{seed}.npy', rowNames=rownames, colNames=range(k), rowDescrip=rownames, datashape = [N, k])
            write_npy(H.get(), f'{args.outputfileprefix}.H.k.{k}.seed.{seed}.npy', rowNames=range(k), colNames=columnnames, rowDescrip=range(k), datashape = [k, M])
          elif args.outputfiletype == 'h5':
            if debug:
              print(f'write_h5...')
            write_h5(W.get(), f'{args.outputfileprefix}.W.k.{k}.seed.{seed}', rowNames=rownames, colNames=range(k), rowDescrip=rownames, datashape = [N, k], comm_world=MPI.COMM_WORLD)
            write_h5(H.get(), f'{args.outputfileprefix}.H.k.{k}.seed.{seed}', rowNames=range(k), colNames=columnnames, rowDescrip=range(k), datashape = [k, M], comm_world=MPI.COMM_WORLD)
            write_h5(W.get(), f'{args.outputfileprefix}.W.k.{k}.seed.{seed}', rowNames=rownames, colNames=range(k), rowDescrip=rownames, datashape = [N, k], comm_world=MPI.COMM_WORLD)
            if debug:
              print(f'after write_h5...')
          elif args.outputfiletype == 'gct':
            H_gct = NP_GCT(data=H.get(), rowNames=list(map(str,range(k))), colNames=columnnames,rowDescrip=list(map(str,range(k))))
            H_gct.write_gct(f'{args.outputfileprefix}.H.k.{k}.seed.{seed}.gct')
            W_gct = NP_GCT(data=W.get(), rowNames=rownames, colNames=list(map(str,range(k))),rowDescrip=rownames)
            W_gct.write_gct(f'{args.outputfileprefix}.W.k.{k}.seed.{seed}.gct')
          else:
            print(f'Sorry, not writing W and H, unless --outputfiletype=npy or h5 or gct.\n')
        else:
          print(f'--keepintermediatefiles not True, not writing out W and H\n')
        togetherstart = time.process_time()
        if debug:
          print(f'{rank}: before cp.asnumpy\n')
        i = cp.argmax(H, axis=0).get()
        if debug:
          print(f'{rank}: after cp.asnumpy\n')
        # uses lots of host memory:
        #together_counts[i[:, None] == i[None, :]] += 1
        H = None
        W = None
        mempool.free_all_blocks()
        if debug:
          print(f'{rank}: before updating together_counts, mempool.used_bytes(): {mempool.used_bytes()}\n')
          print(f'{rank}: before updating together_counts, mempool.total_bytes(): {mempool.total_bytes()}\n')
        # better way would be to have each task update a portion
        # of together_counts, then collect.  For now do it the brutal way.
        #together_counts[i[:, None] == i[None, :]] += 1
        # MxM together_counts.  Do slices at a time instead of
        # single elements
        # rowpad, colpad is the amount in last iteration
        #    myrowcount = (myendrow + 1) - mystartrow
        #    mycolcount = (myendcol + 1) - mystartcol
        #    myVcols = cp.array(V[:,mystartcol:myendcol + 1])
        #    myVrows = cp.array(V[mystartrow:myendrow + 1,:])
        togethermask = i[:, None] == i[None, :]
        if debug:
          print(f'{rank}: togethermask.shape {togethermask.shape}\n')
        togethermask = cp.array(togethermask)
        if debug:
          print(f'{rank}: on GPU, togethermask.shape {togethermask.shape}\n')
        # out of GPU memory:
        #together_counts[togethermask] += 1
        #
        # out of GPU memory:
        #zeroarray = cp.zeros((M,M),dtype=TOGETHERTYPE)
        #cupyx.scatter_add(zeroarray,togethermask,1)
        zeroarray = cp.array(togethermask, dtype=TOGETHERTYPE)
        togethermask = None
        i = None
        if debug:
          print(f'{rank}: after scatter_add zeroarray {zeroarray.shape}\n')
        #together_counts[i[:, None] == i[None, :]] += 1
        together_counts += zeroarray
        #os.system('date')
        # https://numpy.org/doc/stable/reference/arrays.nditer.html#arrays-nditer
        # no  cp.nditer
        #with cp.nditer(together_counts, flags=['multi_index'], op_flags=['readwrite']) as it:
        #  for x in it:
        #    if i[it.multi_index] == i[it.multi_index[1], it.multi_index[0]]:
        #      x = x + 1
        if debug:
          print(f'{rank}: after fast together_counts\n')
        if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
          if debug:
            print(f'{rank}: xxxxxxxxxxxxxxx Elapsed time together_counts update for k={k} seed={seed} : {time.process_time() - togetherstart}\n');
      W = None
      H = None
      WH = None
      if debug:
        print(f'{rank}: end of seed-list for loop\n')
        print(f'{rank}: finished k={k}, seed={seed}\n')
      mempool.free_all_blocks()
    os.chdir(JOBDIR)
    myVrows = None
    myVcols = None
    mempool.free_all_blocks()
    if debug:
      thistime = MPI.Wtime()
      print(f'rank {rank}: finished all k loops: ({thistime - lasttime})\n')
      lasttime = thistime
    if not args.noconsensus:
      consensusstart = time.process_time()
      print(f'{rank}: finished all seed trials for k={k}...\n')
      if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
        # for MPI scatter/gather
        numpy.set_printoptions(threshold=M*M)
        if M <= 80:
          print('consensus matrix shape ({})'.format(together_counts.shape))
          #print(f'together_counts: ({together_counts})\n')
          sys.stdout.write('consensus matrix:')
          for i_index in range(M):
            sys.stdout.write('\n')
            for j_index in range(M):
              sys.stdout.write('{:>2.0f}'.format(together_counts[i_index, j_index]/10))
          sys.stdout.write('\n')
        else:
          print(f'{rank}: consensus matrix larger than 80x80, not printing.\n')
      if debug:
        print(f'{rank}: copying together_counts from GPU to host tchost\n')
      i_counts = together_counts
      tchost = together_counts.get()
      if debug:
        print(f'{rank}: after copying together_counts from GPU to host tchost\n')
      if args.inputfiletype in ('gct','h5') or attributes_dict != None:
        # put this back in after done with npy inputfile
        if args.inputfiletype == 'gct':
          columnnames = gct_data.columnnames
        elif args.inputfiletype == 'h5':
          # h5_data.attributes dset.attrs
          columnnames = h5_data.dset.attrs['column_names']
        elif attributes_dict != None:
          if debug:
            print(f'{rank}: using attributes_dict ({attributes_dict})\n')
          columnnames = attributes_dict['column_names']
        else:
          print(f'should never get here!\n')
        if (rank == 0 or args.parastrategy in ('serial', 'kfactor')) and (args.outputfiletype == 'gct'):
          consensus_gct = NP_GCT(data=tchost, rowNames=columnnames, colNames=columnnames)
          consensus_gct.write_gct('{}.consensus.k.{}.gct'.format(args.outputfileprefix,k))
        elif (rank == 0 or args.parastrategy in ('serial', 'kfactor')) and (args.outputfiletype == 'npy'):
          write_npy(tchost, f'{args.outputfileprefix}.consensus.k.{k}.npy', rowNames=columnnames, colNames=columnnames, rowDescrip=columnnames, datashape = [M, M])
        elif args.outputfiletype == 'h5':
          if debug:
            print(f'{rank}: before write_h5(tchost...\n')
          write_h5(tchost, outputfileprefix=f'{args.outputfileprefix}.consensus.k.{k}', rowNames=columnnames, colNames=columnnames, rowDescrip=columnnames, datashape = [M, M], comm_world=MPI.COMM_WORLD)
          if debug:
            print(f'{rank}: after write_h5(tchost...\n')
        if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
          if debug:
            print(f'{rank}: xxxxxxxxxxxxxxx Elapsed time for k={k}, consensus matrix generation: {time.process_time() - consensusstart}\n');
        # calculate cophenetic correlation constant
        # https://medium.com/@codingpilot25/hierarchical-clustering-and-linkage-explained-in-simplest-way-eef1216f30c5
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        # 1. make M clusters, each with one point
        # 2. compute distance matrix MxM, distance from each cluster to every
        #    other.  Could do just the lower triangle of the symmetric matrix...
        #    different methods to compute cluster-to-cluster distance, default
        #    is nearest point (single).
        # 3. Merge closest two clusters into a new cluster
        # 4. update distance matrix, using new cluster
        # repeat 2 and 3 until only one cluster remains
        print(f'{rank}: before linkage\n')
        cophstart = time.process_time()
        # maybe try replacing with fastcluster:
        # https://pypi.org/project/fastcluster/
        # or:
        # https://docs.rapids.ai/api/cuml/stable/api.html#agglomerative-clustering
        #linkage_mat = scipy.cluster.hierarchy.linkage(tchost)
        #linkage_mat = cupyx.scipy.cluster.hierarchy.linkage(together_counts)
        linkage_mat = fastcluster.linkage(tchost)
        linkageend = time.process_time()
        print(f'{rank}: linkage time: {linkageend - cophstart}\n')
        print(f'{rank}: before pdist\n')
        cdm = scipy.spatial.distance.pdist(tchost)
        #cdm = cupyx.scipy.spatial.distance.pdist(together_counts)
        #tchostfloat = tchost.astype(NUMPYTYPE,)
        #cdm = fastdist.matrix_pairwise_distance(tchostfloat, fastdist.euclidean, "euclidean")
        #cdm = cdm.transpose()
        print(f'{rank}: cdm.shape: {cdm.shape}\n')
        # cdmhost = cdm.get()
        pdistend = time.process_time()
        print(f'{rank}: pdist time: {pdistend - linkageend}\n')
        # coming soon: https://github.com/cupy/cupy/issues/5946
        #cdm = cupyx.scipy.spatial.distance.pdist(together_counts)
        # maybe use fastdist for now:
        # https://pypi.org/project/fastdist/
        print(f'{rank}: before cophenet\n')
        cophenetic_correlation_distance, cophenetic_distance_matrix = scipy.cluster.hierarchy.cophenet(linkage_mat, cdm)
        #cophenetic_correlation_distance, cophenetic_distance_matrix = scipy.cluster.hierarchy.cophenet(linkage_mat, cdmhost)
        cophend = time.process_time()
        print(f'{rank}: cophenet time: {cophend - pdistend}\n')
        #cophenetic_correlation_distance, cophenetic_distance_matrix = cupyx.scipy.cluster.hierarchy.cophenet(linkage_mat, cdm)
        print('k={}, cophenetic_correlation_distance: ({})'.format(k,cophenetic_correlation_distance))
        print(f'{rank}: cophenetic correlation distance calculatioin: {time.process_time() - cophstart}\n');
        # do other postprocessing
        postprocessstart = time.process_time()
        if True or args.postprocess:
          # sort the samples in the consensus matrix for the plot
          # put this back after using npy:
          print(f'{rank}: before DataFrame\n')
          countsdf=pd.DataFrame(tchost, columns=columnnames, index=columnnames)
          print(f'{rank}: before KMeans fit\n')
          kmeans = cluster.KMeans(n_clusters=2).fit(countsdf)
          labels = kmeans.labels_
      
          namedf = pd.DataFrame(labels, index = columnnames)
          sortedNames = namedf.sort_values(0).index
      
          countsdf = countsdf[sortedNames]
          countsdf = countsdf.reindex(sortedNames)
          print(f'{rank}: before to_numpy\n')
          sorted_i_counts = countsdf.to_numpy()
          if rank == 0 and args.outputfiletype == 'gct':
            print(f'{rank}: before sc = NP_GCT\n')
            sc = NP_GCT(data=sorted_i_counts, rowNames=sortedNames, colNames=sortedNames )
            print(f'{rank}: before sc.write_gct\n')
            sc.write_gct('{}.consensus.k.{}.sorted.gct'.format(args.outputfileprefix,k))
          elif rank == 0 and args.outputfiletype == 'npy':
            print(f'{rank}: before sc.write_npy\n')
            write_npy(sorted_i_counts, f'{args.outputfileprefix}.consensus.k.{k}.sorted.npy', rowNames=sortedNames, colNames=sortedNames, rowDescrip=sortedNames, datashape = [M, M])
          elif rank == 0 and args.outputfiletype == 'h5':
            print(f'{rank}: before write_h5 {args.outputfileprefix}.consensus.k.{k}\n')
            print(f'sortedNames {sortedNames}\n')
            write_h5(sorted_i_counts, f'{args.outputfileprefix}.consensus.k.{k}.sorted', rowNames=sortedNames, colNames=sortedNames, rowDescrip=sortedNames, datashape = [M, M], comm_world=MPI.COMM_WORLD)
      
          if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
            print(f'{rank}: before subplots\n')
            fig, ax = plt.subplots()
            fig.set_figwidth(8)
            fig.set_figheight(8)
            print(f'{rank}: before imshow\n')
            im = plt.imshow(sorted_i_counts, cmap='bwr', interpolation='nearest')
        
            ax.set_xticks(np.arange(len(sortedNames)), labels=sortedNames)
            ax.set_yticks(np.arange(len(sortedNames)), labels=sortedNames)
        
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  rotation_mode="anchor")
        
            ax.set_title("Consensus Matrix, k="+str(k))
            fig.tight_layout()
        
            plt.savefig('{}.consensus.k.{}.pdf'.format(args.outputfileprefix,k))  
            with open('{}.cophenetic.txt'.format(args.outputfileprefix), 'w') as file:
                file.write(str(k) + "\t" + str(cophenetic_correlation_distance) + "\n")
        if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
          if debug:
            print(f'{rank}: xxxxxxxxxxxxxxx Elapsed time for k={k}, all postprocessing : {time.process_time() - postprocessstart}\n');
      if rank == 0 or args.parastrategy in ('serial', 'kfactor'):
        if debug:
          print(f'{rank}: xxxxxxxxxxxxxxx Elapsed time for k={k} : {time.process_time() - kstart}\n');
      if debug:
        thistime = MPI.Wtime()
        print(f'rank {rank}: finished consensus matrix: ({thistime - lasttime})\n')
        lasttime = thistime
      i_counts = None
      together_counts = None
      mempool.free_all_blocks()
    else:
      print(f'{rank}: not generating consensus matrix...\n')

except BaseException as e:
  traceback.print_tb(sys.exc_info()[2])
  print(f'{rank}: Unexpected error:', sys.exc_info()[0])
  # cleanup may fail if out of memory or exceeded wallclock limit
  if args.nocleanup == True:
    print(f'{rank}: keeping {JOBDIR}/k.*\n')
  else:
    if rank == 0:
      for kdir in kdirs:
        print(f'{rank}: rmtree of {JOBDIR}/{kdir}\n')
        shutil.rmtree(f'{JOBDIR}/{kdir}')
  raise e

# cleanup may fail if out of memory or exceeded wallclock limit
#if args.keepintermediatefiles == True:
#  print(f'{rank}: keeping {JOBDIR}/k.*\n')
#else:
#    if rank == 0:
#      for kdir in kdirs:
#        print(f'{rank}: rmtree of {JOBDIR}/{kdir}\n')
#        shutil.rmtree(f'{JOBDIR}/{kdir}')

