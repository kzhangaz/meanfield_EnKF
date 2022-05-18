from src import set_up_model
from src import set_up_ensemble

class EnKFmodel(object):
	
	def __init__(self,N,K,control_func,noiselevel,sol_func):
		self.K = K # Dimension of the observed data y
		self.N = N # Dimension of the control 
		self.control_func = control_func
		self.noiselevel = noiselevel
		self.sol_func = sol_func

	set_up_model = set_up_model.set_up_model
	#A,G,observations,u_exact are set

	set_up_ensemble = set_up_ensemble.set_up_ensemble
	# En,initEnsemble,ensembleSize are set

	
	




	