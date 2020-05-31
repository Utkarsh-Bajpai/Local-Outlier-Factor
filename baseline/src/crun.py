import os
import statistics

w, h = 10, 10
Matrix = [[0 for x in range(w)] for y in range(h)]

iindex = 0
for i in range(200, 2000, 200):
    for j in range(1, 11):
        out = os.system("alloptim.exe " + str(i) + str(j))
        Matrix[iindex][j] = out
    iindex += 1

"""

Median = [0 for x in range(w)]

for i in range(0,20):
    Median[i]=statistics.median(Matrix[i])

#yaxis = [2*x*x for x in range(200,4200,200)]
#xaxis = [x for x in range(200,4200,200)]
for i in range(0,20):
    yaxis[i]= yaxis[i]/statistics.median(Matrix[i])

print(Matrix)
print(Median)
print(xaxis)
print(yaxis)

plot(x, y)
"""
