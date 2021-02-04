#question1
import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import norm
mean = 5
var = 1
sd = np.sqrt(var)
x = np.linspace(0,10,10,False)
y = norm.pdf(x,mean,sd)

plt.plot(x,y,'ro')
#question2
plt.figure(figsize=(30,60))

for i in range(11):
    plt.subplot(10,3,i+1)
    z = norm.pdf(x,i,1)
    plt.plot(x,y,'ro')
    plt.plot(x,z,'g--o')
    plt.title("Mean = {0} & Variance = 1".format(i))
