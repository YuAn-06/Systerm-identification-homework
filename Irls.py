import WhiteNoise
import numpy as np
import  matplotlib.pyplot as plt
import Msequence
import seaborn as sns
from scipy.stats import norm
# mu = 0
# sigma = 0.5
# normal =  WhiteNoise.white_noise(500, mu, sigma)
import importlib
importlib.reload(Msequence)
length = 599
t = np.arange(0,50)
g = 0.25*np.e**(-0.5*t) - np.e**(-0.25*t) + 0.75*np.e**(-1*t/6)
u =  Msequence.m_sequence(length)*2-1


v = WhiteNoise.white_noise(length,0,0.02)
z = np.ones(550)

for k in range(49,length):
    g_sum = 0

    for i in range(k-49,k):
        g_mul = g[k-i] * u[i]
        g_sum = g_mul + g_sum
    z[k-49] = g_sum + v[k-49]



Y = z.reshape(550,1)
V = v[49:599]
G=np.ones((550,50))
print()
for i in range(550):
    u1 = u[i:50+i]

    u1 = u1[::-1]

    G[i,0:50] = u1

U = np.dot(np.linalg.pinv(np.dot(np.transpose(G),G)),np.dot(np.transpose(G),Y))

plt.figure(2)
plt.plot(U,c="b",label='Fitted Value')
plt.plot(g,label='Real Value',c='red')
# plt.title('Impulse Response ',fontdict={"size":20})
# plt.xlabel('Times (t) ',fontdict={"size":16})
# plt.legend()
plt.show()
