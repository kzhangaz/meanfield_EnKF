from src import timer
from src.update import update_EnKF
from src.update import update_mean_field

def update_model(self,method):

	stopping = 'discrepancy'
	maxit = 5e3
	# maxit = 5

	if method == 1:
		print('4. EnKF solver with max number of iteration %d' % maxit)
		with timer.Timer('EnKF timer'):
			update_EnKF.update_EnKF(self,maxit,stopping)

	if method == 5:
		Minteracting = self.ensembleSize
		tfin = 10
		with timer.Timer('mean_field timer'):
			update_mean_field.update_mean_field(self,maxit,stopping,Minteracting,tfin)
