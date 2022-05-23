import torch

def moments(En):

	m1 = torch.mean(En,dim=1) 
	m2 = torch.zeros(En.size(dim=0),En.size(dim=0))
	
	for j in range(En.size(dim=1)):
		
		temp = torch.zeros(m2.shape)
		for i,entry in enumerate(En[:,j]):
			temp[i,:] = entry*En[:,j]
		# for i in range(En.size(dim=1)):
		# 	entry = En[i,j]
		# 	temp[i,:] = entry*En[:,j]

		m2 = m2 + temp

	m2 = m2/En.size(dim=1)
	
	return m1,m2

