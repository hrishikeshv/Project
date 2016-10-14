import numpy as np

def gen_a(l, a=1, b=5):
	
	inp = np.linspace(a,b,l)
	out = 2*(np.power(inp,2)) + np.exp(np.pi / inp) * np.sin(2*np.pi*inp)
	return (inp,out)

def gen_b(l, a=1, b=5):
	
	inp = np.linspace(a,b,l)
	out = inp * np.sin(inp) * np.cos(inp)
	return (inp,out)

def gen_c(l, a=1, b=5):
	
	inp = np.linspace(a,b,l)
	out = np.sin(inp**2) - inp/4.0
	return (inp,out)

def gen_d(l, a=1, b=5):
	
	inp = np.linspace(a,b,l)
	out = inp * np.sin(inp**2)
	return (inp,out)
