#!/gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv/bin/python3
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

if __name__ == '__main__':
  print('start of nmf_mgpu.py\n')
  sys.stderr.write('start of nmf_mgpu.py, stderr\n')
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--inputmatrix', dest='inputmatrix', action='store')
  parser.add_argument('-k', '--kfactor', dest='kfactor', action='store')
  parser.add_argument('-j', '--checkinterval', dest='checkinterval', action='store')
  parser.add_argument('-t', '--threshold', dest='threshold', action='store')
  parser.add_argument('-i', '--maxiterations', dest='maxiterations', action='store')
  parser.add_argument('-s', '--seed', dest='seed', action='store')
  args = parser.parse_args()

# read input file, create array on device
V = cp.loadtxt(fname=args.inputmatrix)
print(f'inputarray.shape: ({V.shape})\n')

# seed the PRNG
cp.random.seed(int(args.seed))
  
M = V.shape[1]
N = V.shape[0]
kfactor = int(args.kfactor)
# create H and W random on device
# H = M (inputarray.shape[1]) x k (stored transposed), W = N (inputarra.shape[0]x k
# in the bionmf-gpu code, 
# set_random_values( p_dH, nrows, K, Kp,
# set_random_values( d_W, N, K, Kp,
# should be transposed for H...
# this is over interval [0, 1), might need to add a smidge...
Ht = cp.random.rand(kfactor,M)
# is it better to create the transposed matrix from the start?
H = Ht.transpose()
W = cp.random.rand(N,kfactor)
Wt = W.transpose()
#sys.exit(0)

sys.stderr.write('before update H\n')
# update Ht
# * WH(N,BLMp) = W * pH(BLM,Kp)
# * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
# * Haux(BLM,Kp) = W' * WH(N,BLMp)
# * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W

# WH = W * H
WH = cp.matmul(W, H)

# AUX = V (input matrix) ./ (W*H)
AUX = cp.divide(V, WH)

# WTAux = Wt * AUX
WTAUX = cp.matmul(Wt, AUX)

# how do we get reduced an accumulated ACCWT below?
# sum each column down to a single value...
ACCWT = cp.sum(Wt, axis=1)

# WTAUXDIV = WTAUX ./ ACCWT

WTAUXDIV = cp.divide(WTAUX, ACCWT)

# H = H .* WTAUXDIV
Hnew = cp.multiply(H, WTAUXDIV)

sys.stderr.write('before update W\n')
# update W
# * WH(BLN,Mp) = W(BLN,Kp) * H
# * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
# * Waux(BLN,Kp) = WH(BLN,Mp) * H'
# * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
# generate ACCUMH
ACCH = cp.sum(H, axis=0)

# skip steps up to AUX
# from update_W notes:
#  * Waux(BLN,Kp) = WH(BLN,Mp) * H'
HTAUX = cp.matmul(WH, Ht)

# * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
WWAUX = cp.multiply(W, HTAUX)

Wnew = cp.divide(WWAUX, ACCH)

sys.stderr.write('after update W\n')
