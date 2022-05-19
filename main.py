import math
import torch
from src import control_func as cf
from src import exact_sol as es
# from src import set_up_model as sm
# from src import set_up_ensemble as se
from src import moments
from model import EnKF

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 2 for meanfield
	method = 1
	# a = 0; b = math.pi

	test = 1 # choose the continuous control function
	control_func = cf.control_func(test)

	print("Computing the exact solution...")

	sol_func = es.ExactSolution(test)

	# set up model
	K = 2**8 # Dimension of the observed data
	N = 2**8 # Dimension of the control
	noiselevel = 0.01^2;

	print('2. Setup the model with %d data and level of noise %1.2f\n'%(N,noiselevel))

	# A,G,observations,u_exact = sm.set_up_model(N,K,control_func,noiselevel,sol_func)
	model = EnKF.EnKFmodel(N,K,control_func,noiselevel,sol_func)
	model.set_up_model()

	# set up ensemble
	ensembleSize = 200

	#initEnsemble = 'KL'; % Karhunen-Loeve expansion
	#initEnsemble = 'random'; % Normally distributed around the mean of uexact
	initEnsemble = 'brownian';

	print('3. Ensemble size = %d. Setup the initial ensembles using the %s initialization...\n'%(ensembleSize,initEnsemble))

	model.set_up_ensemble(ensembleSize,initEnsemble)

	# calculate the moments
	# m1,m2 = moments.moments(En)


	
