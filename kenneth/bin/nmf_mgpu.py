#!/gpfs/wolf/trn008/proj-shared/teammesirov/conda_envs/cupyenv/bin/python3
# https://www.pnas.org/doi/10.1073/pnas.0308531101?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed
# https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/s12859-015-0485-4.pdf
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
  sys.stdout.write('start of nmf_mgpu.py, stderr\n')
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
print(f'V: ({V})\n')

# load Michael's ALL_AML_data.txt to comparre with bioinmf.input.txt
#testinput = cp.loadtxt(fname='ALL_AML_data.txt.stripped')
#print(f'inputarray.shape: ({testinput.shape})\n')
#print(f'equa?: ({cp.array_equal(V, testinput)})\n')
#sys.exit(0)

# seed the PRNG
cp.random.seed(int(args.seed))

checkinterval = int(args.checkinterval)
maxiterations = int(args.maxiterations)
threshold = int(args.threshold)
  
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
H = cp.random.rand(kfactor,M)
# is it better to create the transposed matrix from the start?
Ht = H.transpose()
W = cp.random.rand(N,kfactor)
Wt = W.transpose()
#sys.exit(0)
print(f'initial H: ({H})\n')
print(f'initial Ht: ({Ht})\n')
print(f'initial W: ({W})\n')
print(f'initial Wt: ({Wt})\n')

iterationcount = 0
oldclassification = None
sameclassificationcount = 0
while iterationcount < maxiterations:
  iterationcount = iterationcount + 1

  sys.stdout.write('before update H\n')
  # update Ht
  # * WH(N,BLMp) = W * pH(BLM,Kp)
  # * WH(N,BLMp) = Vcol(N,BLMp) ./ WH(N,BLMp)
  # * Haux(BLM,Kp) = W' * WH(N,BLMp)
  # * H(BLM,Kp) = H(BLM,Kp) .* Haux(BLM,Kp) ./ accum_W
  
  # WH = W * H
  print(f' W.shape: {W.shape}, H.shape: {H.shape}\n')
  WH = cp.matmul(W, H)
  print(f'WH: ({WH})\n')

  # AUX = V (input matrix) ./ (W*H)
  AUX = cp.divide(V, WH)
  print(f'AUX: ({AUX})\n')
  
  # WTAux = Wt * AUX
  print(f'Wt: ({Wt})\n')
  print(f' Wt.shape: {Wt.shape}, AUX.shape: {AUX.shape}\n')
  print(f' Wt[0,0] * AUX[0,0]: {Wt[0,0] * AUX[0,0]}\n')
  WTAUX = cp.matmul(Wt, AUX)
  print(f' WTAUX.shape: {WTAUX.shape}\n')
  print(f'WTAUX: ({WTAUX})\n')
  
  # how do we get reduced an accumulated ACCWT below?
  # sum each column down to a single value...
  ACCWT = cp.sum(Wt, axis=1)
  
  # WTAUXDIV = WTAUX ./ ACCWT
  
  print(f' WTAUX.shape: {WTAUX.shape}, ACCWT.shape: {ACCWT.shape}\n')
  print(ACCWT)
  WTAUXDIV = cp.divide(WTAUX.transpose(), ACCWT.transpose())
  
  # H = H .* WTAUXDIV
  Hnew = cp.multiply(H, WTAUXDIV.transpose())
  H = Hnew
  
  sys.stdout.write('before update W\n')
  # update W
  # * WH(BLN,Mp) = W(BLN,Kp) * H
  # * WH(BLN,Mp) = Vrow(BLN,Mp) ./ WH(BLN,Mp)
  # * Waux(BLN,Kp) = WH(BLN,Mp) * H'
  # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
  # generate ACCUMH
  ACCH = cp.sum(H, axis=1)
  
  # skip steps up to AUX
  # from update_W notes:
  #  * Waux(BLN,Kp) = WH(BLN,Mp) * H'
  HTAUX = cp.matmul(WH, Ht)
  
  # * W(BLN,Kp) = W(BLN,Kp) .* Waux(BLN,Kp) ./ accum_h
  WWAUX = cp.multiply(W, HTAUX)
  
  Wnew = cp.divide(WWAUX, ACCH)
  W = Wnew
  
  sys.stdout.write('after update W\n')
  print(f'H: ({H})\n')

  # check classification for Ht
  # * Computes the maximum value of each row in d_A[] and stores its column index in d_Idx[].
  # * That is, returns d_Idx[i], such that:
  # *      d_A[i][ d_Idx[i] ] == max( d_A[i][...] ).
  # *
  # * size_of( d_Idx ) >= height
  # * width <= pitch <= maxThreadsPerBlock
  # * In addition, "pitch" must be a multiple of 'memory_alignment'.
  # Ht is M x k, need array M x 1 to store column index
 
  if iterationcount > checkinterval and divmod(iterationcount, checkinterval)[1] ==0:
    # check classification
    if not oldclassification == None:
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
       if cp.array_equal(oldclassification, newclassification) == True:
         print(f'1. equal: ({cp.array_equal(oldclassification, newclassification)})\n')
         if sameclassificationcount >= threshold:
           print(f'classification unchanged in {sameclassificationcount} trials,breaking.\n')
           break
       else:
         print(f'2. equal: ({cp.array_equal(oldclassification, newclassification)})\n')
         oldclassification = newclassification
         sameclassificationcount = 0
print(f'iterationcount ({iterationcount})\n')
