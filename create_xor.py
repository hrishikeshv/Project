import numpy as np
def xor_data(n,f):
	inp = np.random.uniform(-5,5,(n,f))
	prod = np.prod(inp, axis=-1)
	output = np.ones(n)
	output[prod<0]=0
	return inp,output
