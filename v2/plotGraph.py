from matplotlib import pyplot as plt
import numpy as np

results = map(float, open('trainscoreUCI.txt').readlines())

fdict = {0:'abalone',1:'airfoil',2:'concrete',3:'housing'}
index = 2*np.arange(3)
bw = 0.35
for i in range(4):
	d = results[i*36:(i+1)*36]
	for j in range(3):
		r = d[j*12:(j+1)*12]
		lim = min(np.max(r), 10000)
		plt.ylim(0,lim)
		plt.bar(index, r[::4], bw, alpha=0.8, color='b', label='Uncomp')
		plt.bar(index + bw, r[1::4], bw, alpha=0.8, color='r', label='Comp-Deg5')
		plt.bar(index + 2*bw, r[2::4], bw, alpha=0.8, color='g', label='Comp-Deg10')
		plt.bar(index + 3*bw, r[3::4], bw, alpha=0.8, color='y', label='Comp-Deg15')

		plt.title(fdict[i]+' training score with '+str(j+1)+' hidden layers')
		plt.xticks(index+2*bw, ('sigmoid','relu','linear'))
		plt.legend()
		plt.savefig('graphs/training/'+fdict[i]+'-train_hl'+str(j+1)+'.png')
		plt.clf()
