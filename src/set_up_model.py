from torch import *
from math import pi
import matplotlib as plt

def set_up_model(N,K,control_func,noiselevel,sol_func):
	observations = 0
	L = zeros(N,K)
	L = (K/pi)**2 * (2*eye(K) - \
		diag_embed(ones(K-1),1) - diag_embed(ones(K-1),-1))

	A = L + eye(N,K)
	G = linalg.pinv(A)

	x = linspace(0,pi,N)
	u_exact = t(control_func(x))
	p = mm(G,u_exact)

	if noiselevel > 0:
		gamma = noiselevel*eye(K)
		noise = t(distributions.MultivariateNormal(zeros(K),gamma))
	else:
		gamma = eye(K)
		noise = zeros(K)

	observations = p+noise

	# plot
	fig=plt.figure()
	ax=fig.add_axes([0,0,1,1])
	ax.plot(x,observations,'r')
	ax.legend(labels=('Exact solution','Noisy data'))
	ax1=fig.add_axes([0,0,1,1])
	ax1.plot(x,observations,'r')
	ax1.plot(x,sol_func(x),'k')
	fig.show()

	return A,observations,u_exact