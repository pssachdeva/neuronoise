import numpy as np
from scipy.stats import truncnorm

class LNN:
    def __init__(self, v = None, w = None, N = None, sigmaS = 1., sigmaI = 1., sigmaG = 1., nonlinearity = None, thres = None):
        if v is None:
            self.v = self.struct_weight_maker(N, 1)
        else:
            self.v = v
        
        if w is None:
            self.w = self.struct_weight_maker(N, 1)
        else:
            self.w = w
        
        self.N = self.v.size
        
        if nonlinearity is None:
            self.nonlinearity = self.squared
        elif nonlinearity == 'ReLU':
            self.nonlinearity = self.relu
        elif nonlinearity == 'thres_sq':
            self.nonlinearity = self.trunc_squared
        
        self.sigmaS = sigmaS
        self.sigmaI = sigmaI
        self.sigmaG = sigmaG
        self.thres = thres
    
    @staticmethod
    def struct_weight_maker(N, k):
        # establish base weights
        base = np.arange(1, k+1)
        # determine number of repeats, possibly overcompensating
        repeats = np.ceil(float(N)/k)
        # repeat the weights, and only extract the first N
        weights = np.repeat(base, repeats)[:N]
        return weights
    
    @staticmethod
    def unstruct_weight_maker(N, dist, loc = 0., scale = 1., a = None):
        if dist == 'normal':
            weights = np.random.normal(loc = loc, scale = scale, size = N)
        elif dist == 'lognormal':
            weights = np.random.lognormal(mean = loc, sigma = scale, size = N)
        elif dist == 'truncnorm':
            if a is None:
                a = loc
            weights = truncnorm.rvs(a = a, b = 10**10, loc = loc, scale = scale, size = N)
        return weights
    
    '''Example nonlinearities'''
    def squared(self, l):
        return l**2

    def relu(self, l):
        r = np.copy(l)
        r[r < self.thres] = 0
        return r

    def trunc_squared(self, x):
        if x < self.thres:
            return 0
        else:
            return x**2

    def simulate(self, trials):
        s = np.random.normal(loc = 0., scale = self.sigmaG, size = trials)
        xiI = np.random.normal(loc = 0., scale = self.sigmaI, size = trials)
        private_noise = np.random.normal(loc = 0., scale = self.sigmaG, size = (self.N, trials))
        l = np.outer(self.v, s) + np.outer(self.w, xiI) + private_noise
        r = self.nonlinearity(l)
        return s, l, r
    
    def FI_linear_stage(self):
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)
        sigma_inj = self.sigmaG**2 + self.sigmaI**2 * w2
        fisher_info = (self.sigmaG**2 * v2 + self.sigmaI**2 * (v2 * w2 - vdotw**2))/(self.sigmaG**2 * sigma_inj)
        return fisher_info
    
    def FI_squared_nonlin(self, s):
        vw40 = np.sum(self.v**4)
        vw31 = np.sum(self.v**3 * self.w)
        vw22 = np.sum(self.v**2 * self.w**2)
        vw13 = np.sum(self.v * self.w**3)
        vw04 = np.sum(self.w**4)
        
        F1 = vw40/self.sigmaG**2
        F2 = -2 * s**2 * self.sigmaI**2 * vw31**2/(self.sigmaG**2 + 2 * s**2 * self.sigmaG**2 * self.sigmaI**2 * vw22)
        F3 = (self.sigmaG**2 * self.sigmaI**4 * vw22 + 2 * s**2 * self.sigmaI**6 * (vw22 - 2 * vw13 * vw31))/(self.sigmaG**4 + self.sigmaG**2 * (self.sigmaI**4 * vw04 + 2 * s**2 * self.sigmaI**2 * vw22) + 2 * s**2 * self.sigmaI**6 * (vw04 * vw22 - 2 * vw13**2))
        
        return 4 * s**2 * (F1 + F2 + F3) 
    
    def MI_linear_stage(self):
        v2 = np.sum(self.v**2)
        w2 = np.sum(self.w**2)
        vdotw = np.sum(self.v * self.w)
        ww = np.outer(self.w, self.w)
        wv = np.outer(self.w, self.v)
        vw = np.outer(self.v, self.w)
        vv = np.outer(self.v, self.v)
        
        sigma_inj = self.sigmaG**2 + self.sigmaI**2 * w2
        sigma_stim = self.sigmaG**2 + self.sigmaS**2 * v2
        kappa = sigma_inj * sigma_stim - self.sigmaI**2 * self.sigmaS**2 * vdotw**2
        
        I = -np.log(self.sigmaG) + 0.5 * np.log(kappa/sigma_inj)
        
        # second term
        #A1 = -self.sigmaI**2 * sigma_stim/kappa
        #A2 = self.sigmaI**2 * self.sigmaS**2 * vdotw/kappa
        #A3 = -self.sigmaS**2 * sigma_inj/kappa
        #B1 = -(self.sigmaI**4 * self.sigmaS**2 * vdotw**2)/(kappa * sigma_inj)
        
        #A = 1./(2 * self.sigmaG**2) * (np.identity(self.N) + A1 * ww + A2 * (vw + wv)  + A3 * vv)
        #A_inv = 2 * (self.sigmaG**2 * np.identity(self.N) + self.sigmaS**2 * vv + self.sigmaI**2 * ww)
        
        #B = B1 * ww + A2 * wv + A2 * vw + A3
    
        #I2 = 1./(2. * self.sigmaG**2) * np.trace(np.dot(B, A_inv))
        
        # third term
        #gamma = (2 * self.sigmaG**2 + self.sigmaS**2 * v2) * (self.sigmaG**2 + self.sigmaI**2 * w2) - self.sigmaI**2 * self.sigmaS**2 * vdotw**2
        #D = gamma * (-self.sigmaI**4 * vdotw**2 * ww + self.sigmaI**2 * sigma_inj * vdotw * wv + self.sigmaI**2 * sigma_inj * vdotw * vw - (self.sigmaG**2+ self.sigmaI**2 * w2)**2 * vv)
        #C = kappa * self.sigmaG**2 * sigma_inj * (self.sigmaG**2 * v2 + self.sigmaI**2 * (v2 * w2 - vdotw**2))
        #I3 = -self.sigmaS**2/(2 * kappa) * (self.sigmaG**2 * v2 + self.sigmaI**2 * (v2 * w2 - vdotw**2))
        #np.trace(np.dot(D, A_inv))
        #I4 = -self.sigmaS**2/(2 * self.sigmaG**2 * kappa**2 * sigma_inj) * np.trace(np.dot(D, A_inv))
        return I