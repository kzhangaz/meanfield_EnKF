from pickle import TRUE
from torch import *
from src import covmat
def early_stopping(stopping,i,Mi,Mi1,noise):

	#Discrepancy principle
	if stopping == 'discrepancy':
		if Mi <= linalg.matrix_norm(noise)**2:
			return True
	#Relative error
	if stopping == 'relative':
		tol = 1e-3
		if i > 1:
			if abs(Mi-Mi1) < tol:
				return True

def update_EnKF(self,maxit,stopping):
	print("running update_EnKF...")

	for i in range(maxit):

		self.convergence()

		if i > 0:
			if early_stopping(stopping,i,self.M[i],self.M[i-1],self.noise):
				break
		
		Cup = covmat.covmat(self.En,mm(self.G,self.En))
		Cpp = covmat.covmat(mm(self.G,self.En),mm(self.G,self.En))

	for j in range(self.ensembleSize):
		temp = mm(linalg.inv(Cpp + self.gamma),\
					self.observations - mm(self.G,self.En[:,j]) )
		self.En[:,j] = self.En[:,j] + mm(Cup,temp)

	if (i/maxit * 100) % 10 == 0:
		print(' %d ',i/maxit*100)
	
	self.convergence()
	

	
	return