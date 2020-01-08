import argparse
import h5py
import numpy as np
import time

from nd import LNN, MutualInformation


def main(args):
    # extract arguments
    N = args.N
    n_reps = args.n_reps
    n_samples = args.n_samples
    max_kw = args.max_kw
    sigmaP = args.sigmaP
    sigmaC = args.sigmaC
    sigmaS = args.sigmaS
    results_root = args.results_root
    results_file_tag = args.results_file_tag

    kws = 1 + np.arange(max_kw)
    mis = np.zeros((kws.size, n_reps))

    # iterate over structured weight options
    for kw_idx, kw in enumerate(kws):
        lnn = LNN(N=N, kv=1, kw=kw,
                  nonlinearity='squared',
                  sigmaP=sigmaP, sigmaC=sigmaC, sigmaS=sigmaS)
        # iterate over repetitions
        for rep in range(n_reps):
            # draw samples
            s, l, r = lnn.simulate(n_samples)
            s = s.reshape((n_samples, 1))
            r = r.T
            # estimate and store mutual information
            mi = MutualInformation(X=s, Y=r)
            mis[kw_idx, rep] = mi.mutual_information(k=3)

    # store data
    filename =  \
        'mi_N' + str(N) + \
        '_sigmaP' + str(sigmaP) + \
        '_sigmaC' + str(sigmaC) + \
        '_sigmaS' + str(sigmaS) + \
        '_' + results_file_tag + '.h5'
    results = h5py.File(os.path.join(results_root, filename), 'w')
    results['results'] = mis
    results.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int)
    parser.add_argument('--n_reps', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--max_kw', type=int, default=4)
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
