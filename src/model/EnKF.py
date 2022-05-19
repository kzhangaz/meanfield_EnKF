from src import set_up_model
from src import set_up_ensemble
from update.convergence import convergence

class EnKFmodel(object):
	
	def __init__(self,N,K,control_func,noiselevel,sol_func):
		self.K = K # Dimension of the observed data y
		self.N = N # Dimension of the control 
		self.control_func = control_func
		self.noiselevel = noiselevel
		self.sol_func = sol_func
		self.E = []
		self.R = []
		self.AE = []
		self.AR = []
		self.M = [] # M[i] size: 1 * 1

	set_up_model = set_up_model.set_up_model
	#A,G,observations,u_exact,p,noise,gamma are set
	# u_exact: N * 1
	# noise size: K * 1

	set_up_ensemble = set_up_ensemble.set_up_ensemble
	# En,initEnsemble,ensembleSize,m1,m2 are set
	# En size: N * ensemblrSize
	# m1, m2

	convergence = convergence



	
	




	