#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi - mean(x))^2)
#c = mean(y) - m * mean(x)

def find_mean (var):
    return sum(var)/len(var)

x = [3,7,13,14,16]
y = [2,7,11,17,21]

meanx = find_mean(x)
meany = find_mean(y)

num = 0
den = 0

for i in range(len(x)):
    num = num + ((x[i] - meanx) * (y[i] - meany))
    den = den + pow((x[i] -  meanx), 2)

m = num/den
print(m)

c = round(meany - m * meanx, 1)
print(c)

from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([3,7,13,14,16]).reshape([-1,1])
y = np.array([2,7,11,17,21]).reshape([-1,1])

lin = LinearRegression().fit(x,y)
m = lin.coef_
c = lin.intercept_

print(m,c)