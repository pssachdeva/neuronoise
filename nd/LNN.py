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
        sigmaC=1., sigmaP=1., nonlinearity='squared'
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
            self.nonlinearity_name = 'identity'
            self.nonlinearity = self.identity
        elif nonlinearity == 'squared':
            self.nonlinearity_name = 'squared'
            self.nonlinearity = self.squared
        else:
            raise ValueError('Incorrect nonlinearity choice.')

        # standard deviations of random quantities
        self.sigmaS = sigmaS
        self.sigmaC = sigmaC
        self.sigmaP = sigmaP

    def identity(self, l):
        """Identity function for compatibility with nonlinearity functionality.
        """
        return l

    def squared(self, l):
        """Squares input."""
        return l**2

    def simulate_noise(self, s, n_trials):
        """Simulate the network, given a stimulus.

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
        linear = np.repeat(np.reshape(self.v * s, (self.N, 1)), n_trials, axis=1) \
            + np.outer(self.w, xiC) \
            + xiP
        # nonlinear layer
        output = self.nonlinearity(linear)

        return output, linear

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
        """Calculate the Fisher information after the linear stage given the
        network weights."""
        # useful quantities
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)

        # total noise input
        sigma_noise = self.sigmaP**2 + self.sigmaC**2 * w2

        # calculate fisher information
        numerator = self.sigmaP**2 * v2 + self.sigmaC**2 * (v2 * w2 - vdotw**2)
        denominator = self.sigmaP**2 * sigma_noise
        fisher = numerator / denominator
        return fisher

    def MI_linear_stage(self):
        """Calculate the mutual information after the linear stage given the
        network weights."""
        # useful quantities
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)

        # noise and stimulus scalings
        sigma_noise = self.sigmaP**2 + self.sigmaC**2 * w2
        sigma_stim = self.sigmaP**2 + self.sigmaS**2 * v2
        kappa = sigma_noise * sigma_stim - self.sigmaC**2 * self.sigmaS**2 * vdotw**2

        # calculate mutual information
        mutual = -np.log(self.sigmaP) + 0.5 * np.log(kappa / sigma_noise)
        return mutual

    def FI_nonlinear_stage(self, s):
        """Calculate the Fisher information after a squared nonlinearity."""
        # useful constants
        norm = self.sigmaP**2 + 2 * s**2 * self.v**2 + 2 * self.sigmaC**2 * self.w**2
        vw40 = np.sum(self.v**4/norm)
        vw31 = np.sum(self.v**3 * self.w/norm)
        vw22 = np.sum(self.v**2 * self.w**2/norm)
        vw13 = np.sum(self.v * self.w**3/norm)
        vw04 = np.sum(self.w**4/norm)

        # useful matrices
        VMV = vw40 / (2 * self.sigmaP**2) \
            - vw31**2 * (s**2 * self.sigmaC**2) \
            / (self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22)
        WMW = vw04 / (2 * self.sigmaP**2) \
            - vw13**2 * (s**2 * self.sigmaC**2) \
            / (self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22)
        VMW = vw22 / (2 * self.sigmaP**2) \
            - vw13 * vw31 * (s**2 * self.sigmaC**2) \
            / (self.sigmaP**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaP**2 * vw22)

        LFI = 4 * s**2 * (
            VMV - (2 * self.sigmaC**4) / (1 + 2 * self.sigmaC**4 * WMW) * VMW**2
        )

        return LFI

    def covariance_linear_stage(self):
        """Calculate the covariance matrix after the linear stage."""
        covariance = \
            self.sigmaP**2 * np.identity(self.N) + \
            + self.sigmaC**2 * np.outer(self.w, self.w)
        return covariance

    def covariance_nonlinear_stage(self, s):
        """Calculate the covariance matrix after the nonlinear stage."""
        V = self.v**2
        W = self.w**2
        X = self.v * self.w
        covariance = \
            2 * self.sigmaP**4 * np.identity(self.N) \
            + 4 * self.sigmaP**2 * s**2 * np.diag(V) \
            + 4 * self.sigmaP**2 * self.sigmaC**2 * np.diag(W) \
            + 4 * s**2 * self.sigmaC**2 * np.outer(X, X) \
            + 2 * self.sigmaC**4 * np.outer(W, W)
        return covariance

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
        numerator = (12 * sigmaP**2 + N * sigmaC**2 * (kw**2 - 1))
        denominator = (6 * sigmaP**2 + N * sigmaC**2 * (2 * kw**2 + 3 * kw + 1))
        constant = N / (2 * sigmaP**2)
        fisher = constant * numerator / denominator
        return fisher

    @staticmethod
    def MI_linear_struct(N, kw, sigmaP, sigmaC, sigmaS):
        """Calculate the mutual information of the linear stage, with structured
        weights."""
        numerator = (12 * sigmaP**2 + N * sigmaC**2 * (kw**2 - 1))
        denominator = (6 * sigmaP**2 + N * sigmaC**2 * (2 * kw**2 + 3 * kw + 1))
        constant = N / (2 * sigmaP**2)
        fisher = constant * numerator / denominator
        mutual = 0.5 * np.log(1 + sigmaS**2 * fisher)
        return mutual

    @staticmethod
    def cov2corr(cov):
        """Converts a covariance matrix to a correlation matrix."""
        diagonal = np.diag(1./np.sqrt(np.diag(cov)))
        corr = np.dot(diagonal, np.dot(cov, diagonal))
        return corr
