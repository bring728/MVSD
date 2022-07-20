import numpy as np

a = np.array([1,2,3])

ad = {}
ad['dd'] = a

b = ad['dd']
c = b +0.
c[0] = 3


print(c)
print(ad)