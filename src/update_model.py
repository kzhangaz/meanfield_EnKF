import timer
from update import update_EnKF as upEnKF

def update_model(method,maxit):

	methodC = method

	if method == 1:
		ltype = '-x'
		print('4. EnKF solver with max number of iteration %d' % maxit)
		with timer.Timer('EnKF timer'):
			update_EnKF = 1

	if method == 5:
		dsc=1
