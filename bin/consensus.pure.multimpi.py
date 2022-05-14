#!/gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv/bin/python
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
import heapq

from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.rank
mpi_size = comm.size
rank = mpi_rank
numtasks = mpi_size

#print(f'{mpi_rank}: start\n')

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



############  add local functions
sys.path.append("/gpfs/wolf/trn008/scratch/kenneth/nmf-gpu/wrapper")
from readgct import NP_GCT
sys.path.append("/gpfs/wolf/trn008/scratch/kenneth/nmf-gpu/bin")
from nmf_mgpu_mpi import runnmf
###########

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--parastrategy', dest='parastrategy', action='store', choices=['kfactor', 'inputmatrix', 'serial'])
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
parser.add_argument('-p', '--gpuprogrampath', dest='gpuprogrampath', action='store')
parser.add_argument('-d', '--jobdir', dest='jobdir', action='store')
parser.add_argument('-i', '--inputfile', dest='inputfile', action='store')
parser.add_argument('-m', '--mink', dest='mink', action='store')
parser.add_argument('-e', '--maxk', dest='maxk', action='store')
parser.add_argument('-a', '--startseed', dest='startseed', action='store')
parser.add_argument('-r', '--seeds', dest='seeds', action='store')
parser.add_argument('-j', '--interval', dest='interval', action='store')
parser.add_argument('-t', '--consecutive', dest='consecutive', action='store')
parser.add_argument('-x', '--maxiterations', dest='maxiterations', action='store')
parser.add_argument('-n', '--numtasks', dest='mpitasks', action='store')
parser.add_argument('-o', '--outputfileprefix', dest='outputfileprefix', action='store')
parser.add_argument('-k', '--keepintermediatefiles', dest='keepintermediatefiles', action='store_true')
parser.add_argument('-c', '--nocleanup', dest='nocleanup', action='store_true')
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

V = gct_data.data

M = V.shape[1]
results = []
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
    together_counts = numpy.zeros((M,M))
    for seed in seed_list:
      if debug:
        DEBUGOPTION = '--verbose'
        DEBUGVAL = True
      else:
        DEBUGOPTION = ''
        DEBUGVAL = False
      
      start = time.process_time() 
      WH = runnmf(inputmatrix=V, kfactor=k, checkinterval=int(args.interval), threshold=int(args.consecutive), maxiterations=int(args.maxiterations), seed=seed, debug=DEBUGVAL, comm=comm)
      if debug:
        if mpi_rank == 0:
          print("xxxxxxxxxxxxxxx Elapsed time for k=" + str(k) + ": " + str(time.process_time() - start));

          print("return from runnmf is " + str(WH)) 
      if (not WH):
          continue
      maxrow_list = []
      for mindex in range(M):
        maxrow_list.append([None,0.0])
      
      H = WH[1]
      for row_index, row in enumerate(H[:]):
        if debug:
          if mpi_rank == 0:
            print(f'row: ({row})\n')
        for field_index, field in enumerate(row):
          field_float = field
          if row_index == 0 or maxrow_list[field_index][1] <= field_float:
            if debug:
              print(f'maxrow_list[{field_index}][1] ({maxrow_list[field_index][1]}) <= ({field_float}), setting maxrow_list[{field_index}][0] to ({row_index})\n')
            maxrow_list[field_index][0] = row_index
            maxrow_list[field_index][1] = field_float
        #input_line_index = input_line_index + 1
        
      if debug:
        print(f'maxrow_list: ({maxrow_list})\n')
      # update together_counts
      for i_index in range(M):
        for j_index in range(M):
          if maxrow_list[i_index][0] == maxrow_list[j_index][0]:
            together_counts[i_index, j_index] = together_counts[i_index, j_index] + 1
    if mpi_rank == 0:
      print('finished all seed trials for k={}, calculating cophenetic correlation distance...'.format(k))
    # for MPI scatter/gather
    results.append(together_counts)
    if mpi_rank == 0:
      numpy.set_printoptions(threshold=M*M)
      print('consensus matrix shape ({})'.format(together_counts.shape))
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
    if mpi_rank == 0:
      print('k={}, cophenetic_correlation_distance: ({})'.format(k,cophenetic_correlation_distance))
    
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(8)
    im = plt.imshow(i_counts, cmap='bwr', interpolation='nearest')

    ax.set_xticks(np.arange(len(gct_data.columnnames)), labels=gct_data.columnnames)
    ax.set_yticks(np.arange(len(gct_data.columnnames)), labels=gct_data.columnnames)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    #for i in range(len(gct_data.columnnames)):
    #    for j in range(len(gct_data.columnnames)):
    #        text = ax.text(j, i, i_counts[i, j],
    #                   ha="center", va="center", color="w")

    ax.set_title("Consensus Matrix, k="+str(k))
    fig.tight_layout()

    plt.savefig('{}.consensus.k.{}.pdf'.format(args.outputfileprefix,k))  


  comm.gather(results, root=0)
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
  #os.unlink(JOBDIR + '/bionmf.input.txt')
