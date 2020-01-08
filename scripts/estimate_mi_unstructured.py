"""Estimates the mutual information in the linear-nonlinear network for
unstructured weights, for one choice of mean."""
import argparse
import h5py
import numpy as np
import time

from nd import LNN, MutualInformation

def main(args):
    # extract arguments
    N = args.N
    mu = args.mu
    n_reps = args.n_reps
    n_draws = args.n_draws
    n_samples = args.n_samples
    sigmaP = args.sigmaP
    sigmaC = args.sigmaC
    sigmaS = args.sigmaS
    results_root = args.results_root
    results_file_tag = args.results_file_tag

    mis = np.zeros((n_draws, n_reps))

    # iterate over draws
    for draw in range(n_draws):
        w = 1 + LNN.unstruct_weight_maker(N, 'lognormal', loc=mu, scale=1.),
        lnn = LNN(
            N=N, kv=1, w=w,
            nonlinearity='squared',
            sigmaP=sigmaP, sigmaC=sigmaC, sigmaS=sigmaS)

        for rep in range(n_reps):
            s, l, r = lnn.simulate(n_samples)
            s = s.reshape((n_samples, 1))
            r = r.T
            mi = MutualInformation(X=s, Y=r)
            mis[draw, rep] = mi.mutual_information(k=3)

    filename =  \
        'mi_N' + str(N) + \
        '_sigmaM' + str(sigmaM) + \
        '_sigmaC' + str(sigmaC) + \
        '_sigmaS' + str(sigmaS) + \
        '_mu' + str(mu) + 
        '_' + results_file_tag + '.h5'

    results = h5py.File(os.path.join(results_root, filename), 'w')
    results['results'] = mis
    results.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int)
    parser.add_argument('--mu', type=float)
    # the number of times to estimate the mutual information
    parser.add_argument('--n_reps', type=int, default=50)
    # the number of draws from the lognormal dist
    parser.add_argument('--n_draws', type=int, default=100)
    # the number of samples in each simulated dataset
    parser.add_argument('--n_samples', type=int, default=10000)
    # variances
    parser.add_argument('--sigmaP', type=float, default=1.)
    parser.add_argument('--sigmaC', type=float, default=1.)
    parser.add_argument('--sigmaS', type=float, default=1.)
    # directory to store results in
    parser.add_argument('--results_root', default='')
    # unique tag for the results file
    parser.add_argument('--results_file_tag', default='')
    args = parser.parse_args()

    main(args)