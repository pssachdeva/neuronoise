import argparse
import h5py
import numpy as np
import time
import os

from ksg import *
from LNN import LNN

### load arguments ###
parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int)
parser.add_argument('--mu', type=float)
parser.add_argument('--n_reps', type=int, default=50)
parser.add_argument('--dist_reps', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=10000)
parser.add_argument('--sigmaM', type=float, default=1.)
parser.add_argument('--sigmaC', type=float, default=1.)
parser.add_argument('--sigmaS', type=float, default=1.)
parser.add_argument('--tag', default='')
parser.add_argument('--results_root', default='')
args = parser.parse_args()

N = args.N
mu = args.mu
n_reps = args.n_reps
dist_reps = args.dist_reps
n_samples = args.n_samples
# noise variances
sigmaM = args.sigmaM
sigmaC = args.sigmaC
sigmaS = args.sigmaS
tag = args.tag

mis = np.zeros((n_reps, dist_reps))

# create LNN
for dist_rep in range(dist_reps):
	lnn = LNN(
		N=N, 
		kv=1, w=1 + LNN.unstruct_weight_maker(N, 'lognormal', loc=mu, scale=1.),
		nonlinearity='squared',
		sigmaM=sigmaM, sigmaC=sigmaC, sigmaS=sigmaS
	)
	for rep in range(n_reps):
		t = time.time()
		s, l, r = lnn.simulate(n_samples)
		s = s.reshape((n_samples, 1))
		r = r.T
		mi = MutualInformation(X=s, Y=r)
		mis[dist_rep, rep] = mi.mutual_information(k=3)
		print(time.time() - t)

filename =  tag + 'mi_N' + str(N) + \
			'_sigmaM' + str(sigmaM) + \
			'_sigmaC' + str(sigmaC) + \
			'_sigmaS' + str(sigmaS) + \
			'_mu' + str(mu) + \
			'.h5'
#results = h5py.File('mi_results/' + filename, 'w')
root = args.results_root
results = h5py.File(os.path.join(root, filename), 'w')
results['results'] = mis
results.close()
