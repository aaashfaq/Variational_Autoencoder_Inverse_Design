import permeability

from PIL import Image
from numpy import asarray

def simu(data):
    x=permeability.dvector()
    for i in range(0, data.shape[0]):
    	for j in range(0, data.shape[1]):
    		x.push_back(float(data[i,j,0]))
    p = permeability.getPermeability(x, data.shape[0], data.shape[1], 10000)
    print(' permeability = ', p)
    return p


img = Image.open('pyExample.tif')

data = asarray(img)
print(simu(data))
