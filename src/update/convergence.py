from torch import *

def convergence(En,u_exact,m1,G,p,noise,gamma,iter,E):

	e = En - m1[:,None]
	r = En - u_exact[:,None]

	misfit = mm(G,r) - noise

	E = E.append()
	




	return