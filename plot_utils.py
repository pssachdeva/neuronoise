import h5py
import matplotlib.pyplot as plt
import numpy as np

from LNN import LNN


colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def struct_weight_plot_linear_N(
    Ns, ks, plot, version=1, sigmaP=1., sigmaS=1., sigmaC=1., fax=None
):
    """Plot the Fisher/Mutual informations after the linear layer in a network
    with structured weights, as a function of population size."""
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create data arrays
    data = np.zeros((Ns.size, ks.size))

    # iterate over scales
    for k_idx, k in enumerate(ks):
        # iterate over population sizes
        for N_idx, N in enumerate(Ns):
            lnn = LNN(N=N, sigmaP=sigmaP, sigmaS=sigmaS, sigmaC=sigmaC)

            # calculate fisher information
            if plot == 'FI_linear':
                if version == 1:
                    data[N_idx, k_idx] = lnn.FI_linear_struct(N, k, sigmaP, sigmaC)
                else:
                    data[N_idx, k_idx] = lnn.FI_linear_struct(N, N / k, sigmaP, sigmaC)

            # calculate mutual information
            elif plot == 'MI_linear':
                if version == 1:
                    data[N_idx, k_idx] = lnn.MI_linear_struct(N, k, sigmaP,
                                                              sigmaC, sigmaS)
                else:
                    data[N_idx, k_idx] = lnn.MI_linear_struct(N, N / k, sigmaP,
                                                              sigmaC, sigmaS)

            else:
                raise ValueError('Plot version does not exist.')

        # plot the data, changing the label/colors if necessary
        if version == 1:
            ax.plot(
                Ns, data[:, k_idx],
                label=r'$k_{\mathbf{w}}=%s$' % k,
                linewidth=4,
                color=colors[-k_idx])
        else:
            ax.plot(
                Ns, data[:, k_idx],
                label=r'$k_{\mathbf{w}}=N/%s$' % k,
                linewidth=4,
                color=colors[k_idx])

    ax.set_facecolor('white')
    ax.set_xlabel(r'$N$', fontsize=30)
    ax.tick_params(labelsize=20)
    ax.set_xlim([np.min(Ns), np.max(Ns)])
    lgd = ax.legend(
        loc=2,
        facecolor='white',
        prop={'size': 18},
        handletextpad=0.6,
        handlelength=1.,
        labelspacing=0.27)
    lgd.get_frame().set_edgecolor('k')

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def struct_weight_plot_linear_k(
    Ns, ks, plot, sigmaP=1., sigmaS=1., sigmaC=1., fax=None
):
    """Plot the Fisher/mutual information after the linear layer in a network
    with structured weights as a function of weight diversity."""
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create data arrays
    data = np.zeros((Ns.size, ks.size))

    # iterate over population size
    for N_idx, N in enumerate(Ns):
        # iterate over weight groupings
        for k_idx, k in enumerate(ks):
            lnn = LNN(N=N, sigmaP=sigmaP, sigmaS=sigmaS, sigmaC=sigmaC)

            if plot == 'FI_linear':
                data[N_idx, k_idx] = lnn.FI_linear_struct(N, k, sigmaP, sigmaC)

            elif plot == 'MI_linear':
                data[N_idx, k_idx] = lnn.MI_linear_struct(N, k, sigmaP, sigmaC, sigmaS)

            else:
                raise ValueError('Plot version does not exist.')

        ax.plot(ks, data[N_idx, :], label=r'$N=%s$' % N, linewidth=4, color=colors[N_idx])

    ax.set_facecolor('white')
    ax.set_xlabel(r'$k_{\mathbf{w}}$', fontsize=30)
    ax.set_xlim([np.min(ks), np.max(ks)])
    ax.set_xticks(ks)
    ax.tick_params(labelsize=20)
    lgd = ax.legend(loc=4,
                    facecolor='white',
                    prop={'size': 18},
                    handletextpad=0.4,
                    handlelength=1.,
                    labelspacing=0.27)
    lgd.get_frame().set_edgecolor('k')

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def unstruct_weight_plot_mu(
    Ns, mus, sigma, repetitions, plot, design='lognormal',
    sigmaP=1., sigmaS=1., sigmaC=1., fax=None
):
    """Plot the Fisher/mutual information after the linear layer in a network
    of unstructured weights, averaged over many repetitions, as a function of
    network size."""
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create data arrays
    data = np.zeros((Ns.size, mus.size, repetitions))

    # iterate over population sizes
    for N_idx, N in enumerate(Ns):
        # iterate over weight scales
        for mu_idx, mu in enumerate(mus):
            # iterate over repetitions
            for rep in range(repetitions):
                v = np.ones(N)
                w = 1. + LNN.unstruct_weight_maker(N, design, loc=mu, scale=sigma)
                lnn = LNN(v=v, w=w, sigmaP=sigmaP, sigmaS=sigmaS, sigmaC=sigmaC)

                if plot == 'FI_linear':
                    data[N_idx, mu_idx, rep] = lnn.FI_linear_stage()
                elif plot == 'MI_linear':
                    data[N_idx, mu_idx, rep] = lnn.MI_linear_stage()
                else:
                    raise ValueError('Plot version does not exist.')

                data_means = np.mean(data[N_idx, :, :], axis=1)
                data_stdevs = np.std(data[N_idx, :, :], axis=1)

        ax.plot(
            mus, data_means,
            color=colors[N_idx],
            linestyle='-',
            linewidth=4,
            zorder=10,
            label=r'$N = %s$' % N)
        ax.fill_between(
            mus,
            data_means - data_stdevs,
            data_means + data_stdevs,
            color=colors[N_idx],
            alpha=0.50)

    ax.set_facecolor('white')
    ax.set_xlabel(r'$\mu$', fontsize=30)
    ax.tick_params(labelsize=20)
    lgd = ax.legend(loc=4,
                    facecolor='white',
                    prop={'size': 18},
                    ncol=2,
                    handletextpad=0.4,
                    handlelength=1.,
                    labelspacing=0.27,
                    columnspacing=0.5)
    lgd.get_frame().set_edgecolor('k')

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def struct_weight_plot_nonlinear_N(
    N_max, ks, s=1., version=1, colors=colors,
    sigmaP=1., sigmaC=1., fax=None, linestyle='-'
):
    """Plot the Fisher information"""
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # iterate over diversity values
    for k_idx, k in enumerate(ks):
        Ns = np.arange(k, N_max, k)
        data = np.zeros(Ns.shape)
        # iterate over population sizes
        for N_idx, N in enumerate(Ns):
            # type of structured weights to draw
            if version == 1:
                w = LNN.struct_weight_maker(N, k)
            else:
                w = LNN.struct_weight_maker(N, N/k)

            lnn = LNN(
                v=np.ones(N), w=w,
                sigmaP=sigmaP, sigmaC=sigmaC,
                nonlinearity='squared')
            data[N_idx] = lnn.FI_squared_nonlin(s)

        # plot results depending on the specific version
        if version == 1:
            ax.plot(
                Ns, data,
                label=r'$k=%s$' % k,
                linewidth=4,
                color=colors[-k_idx],
                linestyle=linestyle)
        else:
            ax.plot(
                Ns, data,
                label=r'$k=N/%s$' % k,
                linewidth=4,
                color=colors[k_idx],
                linestyle=linestyle)

    ax.set_facecolor('white')
    ax.set_xlabel(r'$N$', fontsize=30)
    ax.tick_params(labelsize=20)
    lgd = ax.legend(loc=2,
                    ncol=2,
                    facecolor='white',
                    prop={'size': 23},
                    handletextpad=0.4,
                    handlelength=1.,
                    labelspacing=0.27,
                    columnspacing=0.50)
    lgd.get_frame().set_edgecolor('k')

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def struct_weight_plot_nonlinear_k(
    Ns, ks, v=None, s=1., colors=colors,
    sigmaP=1., sigmaC=1., fax=None, linestyle='-'
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    data = np.zeros((Ns.size, ks.size))
    # iterate over population size
    for N_idx, N in enumerate(Ns):
        # iterate over noise diversities
        for k_idx, k in enumerate(ks):
            w = LNN.struct_weight_maker(N, k)
            lnn = LNN(
                v=np.ones(N), w=w,
                nonlinearity='squared',
                sigmaP=sigmaP, sigmaC=sigmaC)
            data[N_idx, k_idx] = lnn.FI_squared_nonlin(s)

        ax.plot(
            ks, data[N_idx, :],
            label=r'$N=%s$' % N,
            linewidth=4,
            color=colors[N_idx],
            linestyle=linestyle)

    ax.set_facecolor('white')
    ax.set_xlabel(r'$k_{\mathbf{w}}$', fontsize=30)
    ax.tick_params(labelsize=20)

    lgd = ax.legend(
        facecolor='white',
        prop={'size': 25},
        handletextpad=0.4,
        handlelength=1.,
        labelspacing=0.27,
        columnspacing=0.50)
    lgd.get_frame().set_edgecolor('k')

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def plot_fisher_nonlinear_2d(
    N, ratios, ks, v=None, s=1., version=1, colors=colors, fax=None
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create stimulus weights
    if v is None:
        v = np.ones(N)

    fishers = np.zeros((ratios.size, ks.size))
    for ratio_idx, ratio in enumerate(ratios):
        sigmaC = 1
        sigmaP = ratio * sigmaC
        for k_idx, k in enumerate(ks):
            if version == 1:
                w = LNN.struct_weight_maker(N, k)
            else:
                w = LNN.struct_weight_maker(N, N/k)

            lnn = LNN(
                v=v, w=w, nonlinearity='squared',
                sigmaP=sigmaP, sigmaC=sigmaC)
            fishers[ratio_idx, k_idx] = lnn.FI_squared_nonlin(s)

        fishers[ratio_idx, :] = fishers[ratio_idx, :] / np.max(fishers[ratio_idx, :])

    ax.grid(False)
    img = ax.imshow(fishers.T, interpolation='spline36')
    ax.tick_params(labelsize=20)
    return img


def plot_fisher_nonlinear_2d_alt(
    N, ratios, ks, v=None, s=1., version=1, colors=colors, fax=None
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create stimulus weights
    if v is None:
        v = np.ones(N)

    fishers = np.zeros((ratios.size, ks.size))
    for ratio_idx, ratio in enumerate(ratios):
        sigmaP = 1.
        sigmaC = ratio * sigmaP
        for k_idx, k in enumerate(ks):
            if version == 1:
                w = LNN.struct_weight_maker(N, k)
            else:
                w = LNN.struct_weight_maker(N, N/k)

            lnn = LNN(
                v=v, w=w, nonlinearity='squared',
                sigmaP=sigmaP, sigmaC=sigmaC)
            fishers[ratio_idx, k_idx] = lnn.FI_squared_nonlin(s)

        fishers[ratio_idx, :] = fishers[ratio_idx, :] / np.max(fishers[ratio_idx, :])

    ax.grid(False)
    img = ax.imshow(fishers.T, interpolation='spline36')
    ax.tick_params(labelsize=20)
    return img


def plot_asymptotic_coefficients(filename, fax=None):
    """Plots the asymptotic coefficients for the."""
    # create plot
    labels = [1, 2, 3]
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    coef_file = h5py.File(filename, 'r')
    sigmaP_vals = list(coef_file)

    ks = np.arange(1, 26)

    for idx, sigmaP in enumerate(sigmaP_vals):
        coefs = coef_file[sigmaP]
        ax.plot(
            ks, coefs,
            linewidth=4,
            label=r'$\sigma_P=%s$' % labels[idx],
            color=colors[-idx - 3])

    lgd = ax.legend(
        facecolor='white',
        prop={'size': 25},
        handletextpad=0.4,
        handlelength=1.2,
        labelspacing=0.27,
        columnspacing=0.50)
    lgd.get_frame().set_edgecolor('k')


def unstruct_weight_plot_nonlinear_mu(
    Ns, mus, sigma, repetitions, design='lognormal',
    sigmaP=1., sigmaS=1., sigmaC=1., nonlinearity='squared', s=1., fax=None,
    colors=colors
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    # create data arrays
    data = np.zeros((Ns.size, mus.size, repetitions))

    # iterate over population sizes
    for N_idx, N in enumerate(Ns):
        # iterate over noise diversities
        for mu_idx, mu in enumerate(mus):
            # iterate over repetitions
            for rep in range(repetitions):
                v = np.ones(N)
                w = 1. + LNN.unstruct_weight_maker(N, design, loc=mu, scale=sigma)
                lnn = LNN(
                    v=v, w=w, nonlinearity=nonlinearity,
                    sigmaP=sigmaP, sigmaS=sigmaS, sigmaC=sigmaC)
                data[N_idx, mu_idx, rep] = lnn.FI_squared_nonlin(s=s)
                data_means = np.mean(data[N_idx, :, :], axis=1)
                data_stdevs = np.std(data[N_idx, :, :], axis=1)
        # plot fisher informations
        ax.plot(
            mus, data_means,
            color=colors[N_idx],
            linestyle='-',
            linewidth=4,
            zorder=10,
            label=r'$N = %s$' % N)
        ax.fill_between(
            mus, data_means - data_stdevs, data_means + data_stdevs,
            color=colors[N_idx],
            alpha=0.50)

    ax.set_facecolor('white')
    ax.set_xlabel(r'$\mu$', fontsize=30)
    ax.tick_params(labelsize=20)

    for spine in ax.spines.values():
        spine.set_edgecolor('k')

    return fig, ax


def plot_unstructured_fisher_nonlinear_2d(
    N, ratios, mus, sigmaC, sigma, repetitions, v=None, s=1., version=1,
    colors=colors, fax=None
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    if v is None:
        v = np.ones(N)

    fishers = np.zeros((ratios.size, mus.size, repetitions))
    avg_fishers = np.zeros((ratios.size, mus.size))

    # iterate over ratios
    for ratio_idx, ratio in enumerate(ratios):
        sigmaP = ratio * sigmaC
        # iterate over noise diversities
        for mu_idx, mu in enumerate(mus):
            # iterate over repetitions
            for rep in range(repetitions):
                w = 1. + LNN.unstruct_weight_maker(N,
                                                   'lognormal',
                                                   loc=mu,
                                                   scale=sigma)
                lnn = LNN(
                    v=v, w=w, nonlinearity='squared',
                    sigmaP=sigmaP, sigmaS=1., sigmaC=sigmaC)

                fishers[ratio_idx, mu_idx, rep] = lnn.FI_squared_nonlin(s)
            avg_fishers[ratio_idx, mu_idx] = np.mean(fishers[ratio_idx, mu_idx, :])
        avg_fishers[ratio_idx, :] = \
            avg_fishers[ratio_idx, :] / np.max(avg_fishers[ratio_idx, :])

    ax.grid(False)
    img = ax.imshow(avg_fishers.T, interpolation='spline36', vmin=0, vmax=1)

    return img, fig, ax


def plot_max_weight_diversity(
    N, sigmaPs, sigmaCs, v=None, s=1., version=1., colors=colors, fax=None
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    if v is None:
        v = np.ones(N)

    max_ks = np.zeros((sigmaPs.size, sigmaCs.size))
    # enumerate over private variances
    for sigmaP_idx, sigmaP in enumerate(sigmaPs):
        # enumerate over common variances
        for sigmaC_idx, sigmaC in enumerate(sigmaCs):
            fishers = np.zeros(10)
            # enumerate over diversities
            for k_idx, k in enumerate(np.arange(1, 11)):
                if version == 1:
                    w = LNN.struct_weight_maker(N, k)
                else:
                    w = LNN.struct_weight_maker(N, N / k)

                lnn = LNN(
                    v=v, w=w,
                    sigmaP=sigmaP, sigmaC=sigmaC,
                    nonlinearity='squared')
                fishers[k_idx] = lnn.FI_squared_nonlin(s)
            max_ks[sigmaP_idx, sigmaC_idx] = np.argmax(fishers) + 1
    img = ax.imshow(max_ks, interpolation='spline36', vmin=1, vmax=10)
    ax.tick_params(labelsize=30)

    return fig, ax, img, max_ks


def plot_max_weight_diversity_unstructured(
    N, reps, sigmaPs, sigmaCs, sigma=1., mus=np.linspace(-1., 2., 11),
    v=None, s=1., fax=None
):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    if v is None:
        v = np.ones(N)

    max_mus = np.zeros((sigmaPs.size, sigmaCs.size))
    # iterate over private variances
    for sigmaP_idx, sigmaP in enumerate(sigmaPs):
        # iterate over common variances
        for sigmaC_idx, sigmaC in enumerate(sigmaCs):
            fishers = np.zeros((11, reps))
            # iterate over scales
            for mu_idx, mu in enumerate(mus):
                # iterate over repetitions
                for rep in range(reps):
                    w = 1. + LNN.unstruct_weight_maker(
                        N, 'lognormal',
                        loc=mu, scale=sigma)
                    lnn = LNN(
                        v=v, w=w,
                        nonlinearity='squared',
                        sigmaP=sigmaP, sigmaC=sigmaC)
                    fishers[mu_idx, rep] = lnn.FI_squared_nonlin(s)
            max_mus[sigmaP_idx, sigmaC_idx] = \
                mus[np.argmax(np.mean(fishers, axis=1)).ravel()[0]]
    img = ax.imshow(max_mus, interpolation='spline36', vmin=-1, vmax=2)
    ax.tick_params(labelsize=30)

    return fig, ax, img, max_mus


def plot_scatter_linear(N, s, kv, kw, n_trials, fax=None, color=None):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    lnn = LNN(N=N, kv=kv, kw=kw, sigmaC=1, sigmaP=1)
    data = lnn.simulate_noise_linear(s=s, n_trials=n_trials)
    esses = np.linspace(5, 35, 100)
    ax.plot(lnn.v[0] * esses, lnn.v[-1] * esses, zorder=10, color='k')
    if color is not None:
        ax.scatter(data[0], data[-1], color=color, label=r'$k_{\mathbf{w}} = %s$' % kw)
    else:
        ax.scatter(data[0], data[-1])

    return fig, ax


def plot_scatter_quadratic(N, s, kv, kw, n_trials, fax=None, color=None):
    # create plot
    if fax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig, ax = fax

    lnn = LNN(N=N, kv=kv, kw=kw, sigmaC=1, sigmaP=1)
    data = lnn.simulate_noise_linear(s=s, n_trials=n_trials)**2
    esses = np.linspace(5, 35, 100)
    ax.plot(lnn.v[0] * esses**2, lnn.v[-1] * esses**2, zorder=10, color='k')
    if color is not None:
        ax.scatter(data[0], data[-1], color=color, label=r'$k_{\mathbf{w}} = %s$' % kw)
    else:
        ax.scatter(data[0], data[-1])

    return fig, ax
