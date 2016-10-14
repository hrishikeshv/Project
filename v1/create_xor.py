import numpy as np
np.random.seed(1234)

def xor_data(n,f):
	inp = np.random.uniform(-1,1,(n,f))
	prod = np.prod(inp, axis=-1)
	output = np.ones(n)
	output[prod<0]=0
	return inp,output
