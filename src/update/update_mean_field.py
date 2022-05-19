from pickle import TRUE
from torch import *
from src import covmat
from src import moments
from update_EnKF import early_stopping

def update_mean_field(self,maxit,stopping,Minteracting,tfin):
	print("running update_mean_field...")


	time = []
	time[0] = 0

	radius = []

	for i in range(maxit):

		eig1 = mm(-(self.m2-mm(self.m1,t(self.m1))),t(self.G))
		eig2 = mm(linalg.pinv(self.gamma),self.G)
		eig = linalg.eig(mm(eig1,eig2))
		if mean(isreal(eig)) != 1:
			eig = real(eig)

		if self.ensembleSize==Minteracting and i==0:
			spectrum = abs(eig)

		radius[i] = max(abs(eig))
		dt = 1/radius[i]

		if time[i]+dt > tfin:
			dt = tfin-time[i]
		if time[i] >= tfin:
			break

		self.convergence()

		if i > 0:
			if early_stopping(stopping,i,self.M[i],self.M[i-1],self.noise):
				break
		
		Cup = covmat.covmat(self.En,mm(self.G,self.En))
		Cpp = covmat.covmat(mm(self.G,self.En),mm(self.G,self.En))

		if Minteracting != self.ensembleSize:
			TempEn = zeros(self.En.size())

		for j in range(self.ensembleSize):
			# xi = noiseLevel.*randn(1,N) # mvnrnd(zeros(1,N),gamma)
			part2 = mm(linalg.inv(self.gamma),self.observations-mm(self.G,self.En[:,j]))

			if Minteracting == self.ensembleSize:
				grad = mm(dt*(-eig1),part2)
				self.En[:,j] = self.En[:,j] + grad # + sqrt(dt)*t(xi);
			else:
				Idx = multinomial(linspace(1,self.ensembleSize,self.ensembleSize),Minteracting)
				m1Idx,m2Idx = moments.moments(self.En[:,Idx])
				part1 = dt* mm( (m2Idx-mm(m1Idx,t(m1Idx))) , t(self.G) )
				grad = mm(part1,part2)
				TempEn[:,j] = self.En[:,j] + grad # + sqrt(dt)*t(xi);
		
		if Minteracting == self.ensembleSize:
			self.m1,self.m2 = moments.moments(self.En)
		else:
			self.En = TempEn
			self.m1,self.m2 = moments.moments(self.En)

		if ((time(i)/tfin) * 100) % 10 == 0:
			print(' %d ',(time(i)/tfin) * 100)
	
	self.convergence()
	
	self.final_plot(i,method=5)
	
	return