import matplotlib.pyplot as plt  
import numpy as np 
x = np.arange(-2,2,0.01) 
y = x**4 - 3 * x**2 - 1*x  + 3

plt.figure(figsize=(20, 10), dpi=50)
plt.plot(x,y, color = "black")
plt.show()