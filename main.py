import math
from src import control_func as cf
from src import exact_sol as es
from src import set_up_model as sm

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 2 for meanfield
	method = 1
	a = 0; b = math.pi

	test = 1 # choose the continuous control function
	control_func = cf.control_func(test)

	print("Computing the exact solution...")

	sol_func = es.ExactSolution(test)

	# set up model
	K = 2**8 # Dimension of the observed data
	N = 2**8 # Dimension of the control
	noiselevel = 0.01^2;

	print('2. Setup the model with %d data and level of noise %1.2f\n'%(N,noiseLevel))

	observations = sm.set_up_model(N,K,control_func,noiselevel,sol_func)
	
	# set up ensemble
	ensembleSize = 200