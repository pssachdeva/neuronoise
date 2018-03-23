import numpy as np
from scipy.stats import truncnorm

class LNN:
	def __init__(self, v = None, w = None, N = None, sigmaS = 1., sigmaC = 1., sigmaM = 1., nonlinearity = None, thres = None):
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
			self.nonlinearity_name = 'squared'
			self.nonlinearity = self.squared
		elif nonlinearity == 'squared':
			self.nonlinearity_name = 'squared'
			self.nonlinearity = self.squared
		elif nonlinearity == 'ReLU':
			self.nonlinearity_name = 'ReLU'
			self.nonlinearity = self.ReLU
		elif nonlinearity == 'squared_ReLU':
			self.nonlinearity_name = 'squared_ReLU'
			self.nonlinearity = self.squared_ReLU
		else:
			raise ValueError('Incorrect nonlinearity choice.')
		
		self.sigmaS = sigmaS
		self.sigmaC = sigmaC
		self.sigmaM = sigmaM
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

	def ReLU(self, l):
		r = np.copy(l)
		r[r < self.thres] = 0
		return r

	def squared_ReLU(self, x):
		if x < self.thres:
			return 0
		else:
			return x**2

	def simulate(self, trials):
		s = np.random.normal(loc = 0., scale = self.sigmaM, size = trials)
		xiI = np.random.normal(loc = 0., scale = self.sigmaC, size = trials)
		private_noise = np.random.normal(loc = 0., scale = self.sigmaM, size = (self.N, trials))
		l = np.outer(self.v, s) + np.outer(self.w, xiI) + private_noise
		r = self.nonlinearity(l)
		return s, l, r
	
	def FI_linear_stage(self):
		v2 = np.sum(self.v**2)
		w2 = np.sum(self.w**2)
		vdotw = np.sum(self.v * self.w)
		sigma_inj = self.sigmaM**2 + self.sigmaC**2 * w2
		fisher_info = (self.sigmaM**2 * v2 + self.sigmaC**2 * (v2 * w2 - vdotw**2))/(self.sigmaM**2 * sigma_inj)
		return fisher_info

	@staticmethod
	def FI_linear_struct(N, kw, sigmaM, sigmaC):
		fisher = N/(2 * sigmaM**2) * (12 * sigmaM**2 + N * sigmaC**2 * (kw**2 - 1))/(6 * sigmaM**2 + N * sigmaC**2 * (2 * kw**2 + 3 * kw + 1))
		return fisher

	@staticmethod
	def MI_linear_struct(N, kw, sigmaM, sigmaC, sigmaS):
		mutual = 0.5 * np.log(1 + sigmaS**2/sigmaM**2 * N - 3 * N**2 * (kw+1)**2 * sigmaC**2 * sigmaS**2/(2 * sigmaM**2 * (N * sigmaC**2 * (kw+1) * (2*kw+1) + 6 * sigmaM**2)))
		return mutual
	
	def FI_nonlinear_stage(self, s):
		if self.nonlinearity_name == 'squared':
			return self.FI_squared_nonlin(s)
		else:
			return ValueError('Nonlinearity not implemented yet.')

	def FI_squared_nonlin(self, s):
		norm = self.sigmaM**2 + 2 * s**2 * self.v**2 + 2 * self.sigmaC**2 * self.w**2
		vw40 = np.sum(self.v**4/norm)
		vw31 = np.sum(self.v**3 * self.w/norm)
		vw22 = np.sum(self.v**2 * self.w**2/norm)
		vw13 = np.sum(self.v * self.w**3/norm)
		vw04 = np.sum(self.w**4/norm)
		
		VMV = vw40/(2 * self.sigmaM**2) - (s**2 * self.sigmaC**2)/(self.sigmaM**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaM**2 * vw22) * vw31**2
		WMW = vw04/(2 * self.sigmaM**2) - (s**2 * self.sigmaC**2)/(self.sigmaM**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaM**2 * vw22) * vw13**2
		VMW = vw22/(2 * self.sigmaM**2) - (s**2 * self.sigmaC**2)/(self.sigmaM**4 + 2 * s**2 * self.sigmaC**2 * self.sigmaM**2 * vw22) * vw13 * vw31

		FI = 4 * s**2 * (VMV - (2 * self.sigmaC**4)/(1 + 2 * self.sigmaC**4 * WMW) * VMW**2)
		return FI

	def FI_squared_nonlin_anal(self, s):
		norm = self.sigmaM**2 + 2 * s**2 * self.v**2 + 2 * self.sigmaC**2 * self.w**2
		vw40 = np.sum(self.v**4/norm)
		vw31 = np.sum(self.v**3 * self.w/norm)
		vw22 = np.sum(self.v**2 * self.w**2/norm)
		vw13 = np.sum(self.v * self.w**3/norm)
		vw04 = np.sum(self.w**4/norm)

		denom = self.sigmaM**4 + self.sigmaM**2 * (self.sigmaC**4 * vw04 + 2 * s**2 * self.sigmaC**2 * vw22) \
				-2*s**2 * self.sigmaC**6 * (vw13**2 - vw04 * vw22)
		numer = self.sigmaM**4 * vw40 - self.sigmaC**4 * self.sigmaM**2 *(vw22**2 - vw04 * vw40) - 2 * s**2 * self.sigmaC**2 * self.sigmaM**2 * (vw31**2 - vw22 * vw40) \
				-2 * s**2 * self.sigmaC**6 * (vw22**3 + vw04 * vw31**2 + vw13**2 * vw40 - vw22 * (2 * vw13 * vw31 + vw04 * vw40))
		return 2 * s**2 * numer/(self.sigmaM**2 * denom)

	def covar_lin(self):
		return self.sigmaM**2 * np.identity(self.N) + self.sigmaS**2 * np.outer(self.v, self.v) + self.sigmaC**2 * np.outer(self.w, self.w)

	def covar_squared_nonlin(self, s):
		V = self.v**2
		W = self.w**2
		X = self.v*self.w
		covar = 2 * self.sigmaM**4 * np.identity(self.N) + 4 * self.sigmaM**2 * s**2 * np.diag(V) + 4 * self.sigmaM**2 * self.sigmaC**2 * np.diag(W) + 4 * s**2 * self.sigmaC**2 * np.outer(X, X) + 2 * self.sigmaC**4 * np.outer(W,W)
		return covar 

	def MI_linear_stage(self):
		v2 = np.sum(self.v**2)
		w2 = np.sum(self.w**2)
		vdotw = np.sum(self.v * self.w)
		ww = np.outer(self.w, self.w)
		wv = np.outer(self.w, self.v)
		vw = np.outer(self.v, self.w)
		vv = np.outer(self.v, self.v)
		
		sigma_inj = self.sigmaM**2 + self.sigmaC**2 * w2
		sigma_stim = self.sigmaM**2 + self.sigmaS**2 * v2
		kappa = sigma_inj * sigma_stim - self.sigmaC**2 * self.sigmaS**2 * vdotw**2
		
		I = -np.log(self.sigmaM) + 0.5 * np.log(kappa/sigma_inj)
		return I