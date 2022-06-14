from mpi4py import MPI
import h5py
import argparse
import sys
import pathlib
import cupy

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfile', dest='inputfile', action='store')
parser.add_argument('-o', '--outputfile', dest='outputfile', action='store')
args = parser.parse_args()
if args.inputfile == None:
  print('--inputfile=<filename> required!\n')
  sys.exit(1)
if args.outputfile == None:
  print('--outputfile=<filename> required!\n')
  sys.exit(1)
inputpath = pathlib.Path(args.inputfile)
if not inputpath.exists():
  print(f'Could not find ({args.inputfile})!\n')
  sys.exit(1)
outputpath = pathlib.Path(args.outputfile)
if outputpath.exists():
  print(f'Output file ({args.outputfile}) exists!\n')
  sys.exit(1)

sys.path.append("/expanse/lustre/projects/ddp242/kenneth/pure/nmf-gpu/wrapper")
from readgct import NP_GCT
gct_data = NP_GCT(args.inputfile)
#print(f'gct_data.data: ({gct_data.data})\n')
#print(f'gct_data.columnnames: ({gct_data.columnnames})\n')
#print(f'gct_data.rownames: ({gct_data.rownames})\n')
#print(f'gct_data.rowdescriptions: ({gct_data.rowdescriptions})\n')
#print(f'gct_data.data.shape: ({gct_data.data.shape})\n')

# https://docs.h5py.org/en/stable/mpi.html
# https://groups.google.com/g/h5py/c/r3nHU7C-tvY
# https://docs.h5py.org/en/latest/high/file.html#version-bounding
rank = MPI.COMM_WORLD.rank
outf = h5py.File(args.outputfile, 'w', driver='mpio', comm=MPI.COMM_WORLD, libver='latest')
datasetname = args.inputfile
#print(f'cupy.asnumpy(gct_data.data): ({cupy.asnumpy(gct_data.data)})\n')
dset = outf.create_dataset(datasetname, data=cupy.asnumpy(gct_data.data), dtype=gct_data.data.dtype)
dset.attrs['column_names'] = gct_data.columnnames
dset.attrs['row_names'] = gct_data.rownames
dset.attrs['row_descriptions'] = gct_data.rowdescriptions
#print(dset.shape)
#print(dset[:,:])
outf.close()
