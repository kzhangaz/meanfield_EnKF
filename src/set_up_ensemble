from sympy import true
from torch import *

def is_sorted(x):
	sorted,_ = diag(x).sort(descending=true)
	return (sorted==x)

def set_up_ensemble(ensembleSize,initEnsemble,u_exact,N,K,A):

	En = zeros(N,ensembleSize)

	if initEnsemble == 'KL':
		beta = 10
		C0 = beta * linalg.pinv(A-eye(N,K))
		
		D,V = linalg.eig(C0);

		if not is_sorted(diag(D)):
			D,I = diag(D).sort(descending=True)
			V = torch.index_select(V,1,I)
			
		D = mm(D , diag(square(randn(K))) )
		
		for j in range(ensembleSize):
			En[:,j] = sqrt(D[j,j] + mean(u_exact)) * V[j,:]

		return En
	
	if initEnsemble == 'random':

		for j in range(N):
			En[j,:] = mean(u_exact) * ones(ensembleSize) + randn(ensembleSize)
		
		return En

	if initEnsemble == 'brownian':

		beta = 10
		C0 = beta * linalg.pinv(A-eye(N,K))
		x = mean(u_exact) * ones(N)
		En = x.repeat(ensembleSize,1) + mm(randn(ensembleSize,N),linalg.cholesky(C0))
		En = t(En)

		return En

	