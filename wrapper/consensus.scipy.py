#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
import gp.data
import pandas
import scipy.cluster.hierarchy
import scipy.spatial.distance

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
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
mpitasks = int(args.mpitasks)
JOBDIR = args.jobdir
# read INPUTFILE as gct file
FO = open(INPUTFILE, 'r')
inputdf = gp.data.GCT(FO)
FO.close()
inputarray = inputdf.to_numpy()
emptydf = inputdf.drop(inputdf.index, axis=0)
FO = open('bionmf.input.txt', 'w')
numpy.savetxt(FO, inputarray, delimiter='\t')
FO.close()
M = inputarray.shape[1]
print('M ({}) from ({})'.format(M,inputarray))
kdirs = []
try:
  for k in range(mink,maxk + 1):
    print('start of loop for k={}'.format(k))
    kdirs.append('k.{}'.format(k))
    os.chdir(JOBDIR)
    os.mkdir('k.{}'.format(k))
    together_counts = numpy.zeros((M,M))
    for seed in seed_list:
      os.chdir(JOBDIR)
      os.mkdir('k.{}/seed.{}'.format(k,seed))
      os.mkdir('k.{}/seed.{}/test'.format(k,seed))
      os.symlink('{}/bionmf.input.txt'.format(JOBDIR),'{}/k.{}/seed.{}/test/bionmf.input.txt'.format(JOBDIR,k, seed))
      os.chdir('k.{}/seed.{}'.format(k,seed))
      cmd = f'jsrun --smpiargs="-gpu" --nrs={mpitasks} --tasks_per_rs=1 --cpu_per_rs=1 --gpu_per_rs=1 --rs_per_host={mpitasks} --bind=rs {GPUPROGRAMPATH}  test/bionmf.input.txt  -k {k}  -j 10  -t 40  -i 2000 -s {seed} > mpi.out 2>mpi.err'
      os.system(cmd)
      maxrow_list = []
      for mindex in range(M):
        maxrow_list.append([None,0.0])
      input_line_index = -1
      for input_line in open('test/{}_H.txt'.format(os.path.split('bionmf.input.txt')[1])).readlines():
        input_line_index = input_line_index + 1
        fields = input_line.split()
        if fields[0] == 'Name':
          continue
        for field_index, field in enumerate(fields):
          if field_index == 0:
            continue
          field_float = float(field.strip())
          if maxrow_list[field_index - 1][1] <= field_float:
            maxrow_list[field_index - 1][1] = field_float
            maxrow_list[field_index - 1][0] = input_line_index
      # update together_counts
      for i_index in range(M):
        for j_index in range(M):
          if maxrow_list[i_index][0] == maxrow_list[j_index][0]:
            together_counts[i_index, j_index] = together_counts[i_index, j_index] + 1
    print('finished all seed trials for k={}, calculating cophenetic correlation distance...'.format(k))
    os.chdir(JOBDIR + '/k.{}'.format(k))
    if args.keepintermediatefiles == True:
      print('keeping ' + JOBDIR + '/k.{}'.format(k))
    else:
      print('rmtree of ' + JOBDIR + '/k.{}'.format(k))
      shutil.rmtree(JOBDIR + '/k.{}'.format(k))
    # together_counts is a square matrix where `together_counts[i,j]` is the
    #number of times sample `i` clusters with sample `j`
    numpy.set_printoptions(threshold=M*M)
    print('consensus matrix shape ({})'.format(together_counts.shape))
    sys.stdout.write('consensus matrix:')
    for i_index in range(M):
      sys.stdout.write('\n')
      for j_index in range(M):
        sys.stdout.write('{:>2.0f}'.format(together_counts[i_index, j_index]/10))
    sys.stdout.write('\n')
    # need to write consensus matrix file <outputfileprefix>.consensus.k.<k>.gct
    consensusdf = pandas.DataFrame(data=together_counts)
    indexlist = []
    for cl in emptydf.columns.values:
      indexlist.append((cl, 'na'))
    consensusdf.set_axis(emptydf.columns.values, axis='columns', inplace=True)
    consensusdf.set_axis(indexlist, axis='index', inplace=True)
    consensusdf = emptydf.append(consensusdf)
    gp.data.write_gct(consensusdf,'{}/{}.consensus.k.{}.gct'.format(JOBDIR,args.outputfileprefix,k))
    linkage_mat = scipy.cluster.hierarchy.linkage(together_counts)
    cdm = scipy.spatial.distance.pdist(together_counts)
    cophenetic_correlation_distance, cophenetic_distance_matrix = scipy.cluster.hierarchy.cophenet(linkage_mat, cdm)
    print('k={}, cophenetic_correlation_distance: ({})'.format(k,cophenetic_correlation_distance))
except:
  print("Unexpected error:", sys.exc_info()[0])
  if args.nocleanup == True:
    print('keeping ' + JOBDIR + '/k*')
    print('keeping ' + JOBDIR + '/bionmf.input.txt')
  else:
    os.unlink(JOBDIR + '/bionmf.input.txt')
    for kdir in kdirs:
      print('rmtree of ' + JOBDIR + '/k.{}'.format(k))
      shutil.rmtree(JOBDIR + '/k.{}'.format(k))
  raise

if args.keepintermediatefiles == True:
  print('keeping ' + JOBDIR + '/bionmf.input.txt')
else:
  print('unlink of ' + JOBDIR + '/bionmf.input.txt')
  os.unlink(JOBDIR + '/bionmf.input.txt')
