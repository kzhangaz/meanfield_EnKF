from torch import mean
from torch import zeros
from torch import t
from torch import mm

def covmat(X,Y):
	ensembleSize = X.size(dim=1)
	N = X.size(dim=0)

	Xbar = mean(X,dim=1)
	Ybar = mean(Y,dim=1);

	C = zeros(N)
	for j in range(ensembleSize):
		C = C + mm(X[:,j]-Xbar,t(X[:,j]-Xbar))

	C = C/ensembleSize
	return C
