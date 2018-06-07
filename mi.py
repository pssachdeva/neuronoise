import argparse
import h5py
import numpy as np
import time

from ksg import *
from LNN import LNN

### load arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--n_reps', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=100000)
parser.add_argument('--sigmaM', type=float, default=1.)
parser.add_argument('--sigmaC', type=float, default=1.)
parser.add_argument('--sigmaS', type=float, default=1.)
parser.add_argument('--tag', default='')
args = parser.parse_args()

N = args.N
n_reps = args.n_reps
n_samples = args.n_samples
sigmaM = args.sigmaM
sigmaC = args.sigmaC
sigmaS = args.sigmaS
tag = args.tag

kws = np.array([1., 2., 3., 4.])
mis = np.zeros((kws.size, n_reps))

# create LNN
for kw_idx, kw in enumerate(kws):
	lnn = LNN(
		N=N, 
		kv=1, kw=kw,
		nonlinearity='squared',
		sigmaM=sigmaM, sigmaC=sigmaC, sigmaS=sigmaS
	)
	for rep in range(n_reps):
		s, l, r = lnn.simulate(n_samples)
		s = s.reshape((n_samples, 1))
		r = r.T
		mi = MutualInformation(X=s, Y=r)
		mis[kw_idx, rep] = mi.mutual_information(k=3)

filename =  'mi_N' + str(N) + \
			'_sigmaM' + str(sigmaM) + \
			'_sigmaC' + str(sigmaC) + \
			'_sigmaS' + str(sigmaS) + \
			'_' + tag + \
			'.h5'
results = h5py.File('mi_results/' + filename, 'w')
results['results'] = mis
results.close()
