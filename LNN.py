import numpy as np
from scipy.stats import truncnorm


class LNN:
    """Class for a linear-nonlinear network with common noise. Simulates data
    or calculates Fisher information quantities given synaptic weights.

    Attributes
    ----------
    N : int
        The number of neurons in the network.

    v : nd-array
        The weights for the stimulus.

    w : nd-array
        The weights for the common noise.

    sigmaS : float
        The standard deviation of the stimulus.

    sigmaC : float
        The standard deviation of the common noise.

    sigmaP : float
        The standard deviation of the private noise.

    nonlinearity : function
        The type of nonlinearity to use.
    """
    def __init__(
        self, v=None, w=None, kv=None, kw=None, N=None, sigmaS=1.,
        sigmaC=1., sigmaP=1., nonlinearity=None
    ):
        # choose stimulus weights
        if v is None:
            # k is provided, default kv = 1
            if kv is None:
                self.v = self.struct_weight_maker(N, 1)
            else:
                self.v = self.struct_weight_maker(N, kv)
        else:
            # weights are provided
            self.v = v

        # choose common noise weights
        if w is None:
            # k is provided, default kw = 1
            if kw is None:
                self.w = self.struct_weight_maker(N, 1)
            else:
                self.w = self.struct_weight_maker(N, kw)
        else:
            # weights are provided
            self.w = w

        # population size
        self.N = self.v.size

        # choose nonlinearity: default, 'squared'
        if nonlinearity is None:
            self.nonlinearity_name = 'squared'
            self.nonlinearity = self.squared
        elif nonlinearity == 'squared':
            self.nonlinearity_name = 'squared'
            self.nonlinearity = self.squared
        elif nonlinearity == 'ReLU':
            self.nonlinearity_name = 'ReLU'
            self.nonlinearity = self.ReLU
        else:
            raise ValueError('Incorrect nonlinearity choice.')

        # standard deviations of random quantities
        self.sigmaS = sigmaS
        self.sigmaC = sigmaC
        self.sigmaP = sigmaP

    # Possible nonlinearities
    def squared(self, l):
        """Squares input."""
        return l**2

    def ReLU(self, l):
        """Applies a rectified linear unit to input."""
        r = np.copy(l)
        r[r < self.thres] = 0
        return r

    def simulate_noise_linear(self, s, n_trials):
        """Simulate the linear component of the network, given a stimulus.

        Parameters
        ----------
        s : float
            The stimulus value.

        n_trials : int
            The number of trials to simulate.

        Returns
        -------
        output : nd-array, shape (trials,)
            The output of the linear layer over the trials.
        """
        # common noise
        xiC = np.random.normal(loc=0, scale=self.sigmaC, size=(1, n_trials))
        # private noise
        xiP = np.random.normal(loc=0, scale=self.sigmaP, size=(self.N, n_trials))

        # linear layer output
        output = np.repeat(np.reshape(self.v * s, (self.N, 1)), n_trials, axis=1) \
            + np.outer(self.w, xiC) \
            + xiP

        return output

    def simulate(self, n_trials):
        """Simulate the network.

        Parameters
        ----------
        n_trials : int
            The number of trials to simulate.

        Returns
        -------
        output : nd-array, shape (trials,)
            The outputs of the network over the trials.
        """
        # stimulus
        s = np.random.normal(loc=0., scale=self.sigmaS, size=n_trials)
        # common noise
        xiC = np.random.normal(loc=0., scale=self.sigmaC, size=n_trials)
        # private noise
        xiP = np.random.normal(loc=0., scale=self.sigmaP, size=(self.N, n_trials))

        # linear layer
        linear = np.outer(self.v, s) + np.outer(self.w, xiC) + xiP
        # nonlinear layer
        output = self.nonlinearity(linear)

        return s, linear, output

    def FI_linear_stage(self):
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)
        sigma_inj = self.sigmaP**2 + self.sigmaC**2 * w2
        fisher_info = (self.sigmaP**2 * v2 + self.sigmaC**2 * (v2 * w2 - vdotw**2))/(self.sigmaP**2 * sigma_inj)
        return fisher_info

    def FI_nonlinear_stage(self, s):
        if self.nonlinearity_name == 'squared':
            return self.FI_squared_nonlin(s)
        else:
            return ValueError('Nonlinearity not implemented yet.')

    def FI_squared_nonlin(self, s):
        norm = self.sigmaP**2 + 2 * s**2 * self.v**2 + 2 * self.sigmaC**2 * self.w**2
        vw40 = np.sum(self.v**4/norm)
        vw31 = np.sum(self.v**3 * self.w/norm)
        vw22 = np.sum(self.v**2 * self.w**2/norm)
        vw13 = np.sum(self.v * self.w**3/norm)
        vw04 = np.sum(self.w**4/norm)

        VMV = vw40/(2 * self.sigmaP**2) - (s**2 * self.sigmaC**2)/(self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22) * vw31**2
        WMW = vw04/(2 * self.sigmaP**2) - (s**2 * self.sigmaC**2)/(self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22) * vw13**2
        VMW = vw22/(2 * self.sigmaP**2) - (s**2 * self.sigmaC**2)/(self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22) * vw13 * vw31

        FI = 4 * s**2 * (VMV - (2 * self.sigmaC**4)/(1 + 2 * self.sigmaC**4 * WMW) * VMW**2)
        return FI

    def FI_squared_nonlin_anal(self, s):
        norm = self.sigmaP**2 + 2 * s**2 * self.v**2 + 2 * self.sigmaC**2 * self.w**2
        vw40 = np.sum(self.v**4/norm)
        vw31 = np.sum(self.v**3 * self.w/norm)
        vw22 = np.sum(self.v**2 * self.w**2/norm)
        vw13 = np.sum(self.v * self.w**3/norm)
        vw04 = np.sum(self.w**4/norm)

        denom = self.sigmaP**4 + self.sigmaP**2 * (self.sigmaC**4 * vw04 + 2 * s**2 * self.sigmaC**2 * vw22) \
                -2*s**2 * self.sigmaC**6 * (vw13**2 - vw04 * vw22)
        numer = self.sigmaP**4 * vw40 - self.sigmaC**4 * self.sigmaP**2 *(vw22**2 - vw04 * vw40) - 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * (vw31**2 - vw22 * vw40) \
                -2 * s**2 * self.sigmaC**6 * (vw22**3 + vw04 * vw31**2 + vw13**2 * vw40 - vw22 * (2 * vw13 * vw31 + vw04 * vw40))
        return 2 * s**2 * numer/(self.sigmaP**2 * denom)

    def covar_lin(self):
        return self.sigmaP**2 * np.identity(self.N) + self.sigmaS**2 * np.outer(self.v, self.v) + self.sigmaC**2 * np.outer(self.w, self.w)

    def covar_squared_nonlin(self, s):
        V = self.v**2
        W = self.w**2
        X = self.v*self.w
        covar = 2 * self.sigmaP**4 * np.identity(self.N) + 4 * self.sigmaP**2 * s**2 * np.diag(V) + 4 * self.sigmaP**2 * self.sigmaC**2 * np.diag(W) + 4 * s**2 * self.sigmaC**2 * np.outer(X, X) + 2 * self.sigmaC**4 * np.outer(W,W)
        return covar

    def covar_lin_stim(self):
        covar = self.sigmaP**2 * np.identity(self.N) + self.sigmaC**2 * np.outer(self.w, self.w) + self.sigmaS**2 * np.outer(self.v, self.v)
        return covar

    def correlation_matrix(self, s):
        if self.nonlinearity_name == 'identity':
            return
        elif self.nonlinearity_name == 'squared':
            V = self.v**2
            W = self.w**2
            X = self.v*self.w
            covar = 2 * self.sigmaP**4 * np.identity(self.N) + 4 * self.sigmaP**2 * s**2 * np.diag(V) + 4 * self.sigmaP**2 * self.sigmaC**2 * np.diag(W) + 4 * s**2 * self.sigmaC**2 * np.outer(X, X) + 2 * self.sigmaC**4 * np.outer(W, W)
            inv_diag = np.diag(np.sqrt(1./np.diag(covar)))
            corr = np.dot(inv_diag, np.dot(covar, inv_diag))
        return corr

    def MI_linear_stage(self):
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)

        sigma_inj = self.sigmaP**2 + self.sigmaC**2 * w2
        sigma_stim = self.sigmaP**2 + self.sigmaS**2 * v2
        kappa = sigma_inj * sigma_stim - self.sigmaC**2 * self.sigmaS**2 * vdotw**2

        I = -np.log(self.sigmaP) + 0.5 * np.log(kappa/sigma_inj)
        return I

    @staticmethod
    def struct_weight_maker(N, k):
        """Creates a vector of "structured" weights.

        Parameters
        ----------
        N : int
            The number of neurons in the network.

        k : int
            The number of groupings.

        Returns
        -------
        weights : nd-array
            The structured weights.
        """
        # establish base weights
        base = np.arange(1, k + 1)
        # determine number of repeats, possibly overcompensating
        repeats = np.ceil(float(N) / k)
        # repeat the weights, and only extract the first N
        weights = np.repeat(base, repeats)[:N]

        return weights

    @staticmethod
    def unstruct_weight_maker(N, dist, loc=0., scale=1., a=None):
        """Creates a vector of "unstructured weights.

        Parameters
        ----------
        N : int
            The number of neurons in the network.

        dist : string
            The distribution to draw the parameters from.

        loc : float
            The location parameter of the distribution.

        scale : float
            The scale parameter of the distribution.

        a : float or None
            Additional parameter for truncated normal distribution.

        Returns
        -------
        weights : nd-array
            The unstructured weights.
        """
        # draw weights according to distribution
        if dist == 'normal':
            weights = np.random.normal(loc=loc, scale=scale, size=N)

        elif dist == 'lognormal':
            weights = np.random.lognormal(mean=loc, sigma=scale, size=N)

        elif dist == 'truncnorm':
            if a is None:
                a = loc
            weights = truncnorm.rvs(a=a, b=10**10, loc=loc, scale=scale, size=N)

        return weights

    @staticmethod
    def FI_linear_struct(N, kw, sigmaP, sigmaC):
        """Calculate the Fisher information of the linear stage, with structured
        weights."""
        fisher = N/(2 * sigmaP**2) * (12 * sigmaP**2 + N * sigmaC**2 * (kw**2 - 1))/(6 * sigmaP**2 + N * sigmaC**2 * (2 * kw**2 + 3 * kw + 1))
        return fisher

    @staticmethod
    def MI_linear_struct(N, kw, sigmaP, sigmaC, sigmaS):
        """Calculate the mutual information of the linear stage, with structured
        weights."""
        mutual = 0.5 * np.log(1 + sigmaS**2/sigmaP**2 * N - 3 * N**2 * (kw+1)**2 * sigmaC**2 * sigmaS**2/(2 * sigmaP**2 * (N * sigmaC**2 * (kw+1) * (2*kw+1) + 6 * sigmaP**2)))
        return mutual
