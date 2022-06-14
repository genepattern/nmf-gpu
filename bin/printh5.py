from mpi4py import MPI
import h5py
import argparse
import sys
import pathlib
import cupy

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', dest='inputfile', action='store')
args = parser.parse_args()
if args.inputfile == None:
  print('--inputfile=<filename> required!\n')
  sys.exit(1)
inputpath = pathlib.Path(args.inputfile)
if not inputpath.exists():
  print(f'Could not find ({args.inputfile})!\n')
  sys.exit(1)

# https://docs.h5py.org/en/stable/mpi.html
# https://groups.google.com/g/h5py/c/r3nHU7C-tvY
# https://docs.h5py.org/en/latest/high/file.html#version-bounding
rank = MPI.COMM_WORLD.rank
inf = h5py.File(args.inputfile, 'r', driver='mpio', comm=MPI.COMM_WORLD, libver='latest')
datasetname = args.inputfile
print(f'inf.keys(): ({inf.keys()})\n')
dset = inf[list(inf.keys())[0]]
print(dset[:,:])
print(f'dset.attrs["column_names"]: ({dset.attrs["column_names"]})\n')
print(f'dset.attrs["row_names"]: ({dset.attrs["row_names"]})\n')
print(f'dset.attrs["row_descriptions"]: ({dset.attrs["row_descriptions"]})\n')
inf.close()
