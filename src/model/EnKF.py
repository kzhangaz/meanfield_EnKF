from src import set_up_model
from src import set_up_ensemble
from src.update.convergence import convergence
from src import update_model

class EnKFmodel(object):
	
	def __init__(self,N,K,control_func,noiselevel,sol_func):
		self.K = K # Dimension of the observed data y
		self.N = N # Dimension of the control 
		self.control_func = control_func
		self.noiselevel = noiselevel # scalar
		self.sol_func = sol_func

		self.E = []
		self.R = []
		self.AE = []
		self.AR = []
		self.M = [] # M[i] size: 1 * 1

	set_up_model = set_up_model.set_up_model
	#A,G,observations,u_exact,p,noise,gamma are set
	# G (y = G*u): K * N
	# p: K
	# obsevations: K
	# u_exact: N
	# noise size: K
	# gamma (cov of noise distribution):  K * K

	set_up_ensemble = set_up_ensemble.set_up_ensemble
	# En,initEnsemble,ensembleSize,m1,m2 are set
	# ensembleSize: J=200
	# En size: N * J
	# m1: N
	# m2: 

	convergence = convergence

	update_model = update_model.update_model



	
	




	