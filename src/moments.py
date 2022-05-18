from torch import *

def moments(En):

	m1 = mean(En,dim=1)
	m2 = zeros(En.size(dim=0),En.size(dim=0))
	
	for j in range(En.size(dim=1)):
		m2 = m2 + mm(En[:,j],t(En[:,j]))

	m2 = m2/En.size(dim=1)
	
	return m1,m2

