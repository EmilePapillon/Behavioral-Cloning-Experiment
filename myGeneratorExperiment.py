import numpy as np
def myGenerator():
	res= np.zeros((3,2,5))
	yield res
	yield res+5

gen=myGenerator()
print(len(gen))
for item in gen:
	print(item.shape) 

