import math
from src import control_func as cf

if __name__ == "__main__":
	
	# choose method: 1 for EnKF, 2 for meanfield
	method = 1
	a = 0; b = math.pi

	test = 1 # choose the continuous control function
	control_func = cf.control_func(test)

	print("Computing the exact solution...")
	
