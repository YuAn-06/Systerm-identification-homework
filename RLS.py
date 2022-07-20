# Copyright (C) 2021 #
# @Time    : 2021/11/16 18:03
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : RLS.py
# @Software: PyCharm
import Msequence
import numpy as np
import WhiteNoise
import importlib
import matplotlib.pyplot as plt
importlib.reload(Msequence)
importlib.reload(WhiteNoise)
def RLS():
    P = 100000*np.eye(4,4,dtype=float)

    theta = 0.000001 * np.ones([4,1],dtype=float)

    u = Msequence.m_sequence(l=500)
    v = WhiteNoise.white_noise(500,0,0.1)
    y = np.zeros(1000,dtype=np.float)

    # a1 = np.zeros(499)
    # a2 = np.zeros(499)
    # b1 = np.zeros(499)
    # b2 = np.zeros(499)
    #
    # k1 = np.zeros(499)
    # k2 = np.zeros(499)
    # k3 = np.zeros(499)
    # k4 = np.zeros(499)

    # p11 = np.zeros(499)
    # p12 = np.zeros(499)
    # p13 = np.zeros(499)
    # p14 = np.zeros(499)
    # p22 = np.zeros(499)
    # p23 = np.zeros(499)
    # p24 = np.zeros(499)
    # p33 = np.zeros(499)
    # p34 = np.zeros(499)
    # p44 = np.zeros(499)
    # p = np.zeros((10,499))
    for i in range(1,499):

        y[i+1] = 1.5*y[i]-0.7*y[i-1] + u[i] + 0.5*u[i-1] + v[i+1]

        h = np.array([-y[i], -y[i-1],u[i],u[i-1] ]).reshape(4,1)

        K =  np.dot(P,h)*(1/(np.dot(np.dot(h.T,P),h)+1))

        # k1[i] = K[0]
        # k2[i] = K[1]
        # k3[i] = K[2]
        # k4[i] = K[3]
        theta = theta + np.dot(K,(y[i+1]-np.dot(np.transpose(h),theta)))
        # a1[i] =theta[0]
        # a2[i] = theta[1]
        # b1[i] = theta[2]
        # b2[i] = theta[3]
        P = P-np.dot(np.dot(K,np.transpose(h)),P)
        p[:,i] = [P[0,0],P[0,1],P[0,2],P[0,3],P[1,1],P[1,2],P[1,3],P[2,2],P[2,3],P[3,3]]
    # plt.figure(figsize=(8,6),dpi=600)
    # plt.tick_params(labelsize=15)
    # # plt.plot(k1,label="$K_1$")
    # plt.plot(k4,label="$K_4$")
    # plt.xlim(-5,300)
    # plt.legend()
    # plt.tight_layout()
    # # plt.plot(k3,label="$K_3$")
    # # plt.plot(k4,label="$K_4$")
    # plt.savefig("k4.png")

    # plt.show()

    # plt.figure(1,figsize=(8,6),dpi=600)
    # plt.title('$a_1=-1.5$',fontdict={"size":20})
    # plt.plot(a1,c='orange',label='$a_1$')
    # plt.tick_params(labelsize=15)
    # plt.xlabel('Epochs',fontdict={"size":18})
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig("a1.jpg")
    #
    # plt.figure(2,figsize=(8,6),dpi=600)
    # plt.tick_params(labelsize=15)
    #
    # plt.plot(a2,c='purple',label='$a_2$')
    # plt.title('$a_2=0.7$', fontdict={"size":20})
    # plt.xlabel('Epochs',fontdict={"size":18})
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("a2.jpg")
    #
    # plt.figure(3,figsize=(8,6),dpi=600)
    # plt.tick_params(labelsize=15)
    #
    # plt.plot(b1,c='blue',label='$b_1$')
    # plt.xlabel('Epochs',fontdict={"size":18})
    # plt.legend()
    # plt.title('$b_1=1$', fontdict={"size":20})
    # plt.tight_layout()
    # plt.savefig("b1.jpg")
    #
    # #
    # plt.figure(4,figsize=(8,6),dpi=600)
    # plt.tick_params(labelsize=15)
    # plt.plot(b2, label="$b_2$", c='brown')
    #
    # plt.title('$b_2=0.5$', fontdict={"size":20})
    # plt.xlabel('Epochs', fontdict={"size": 18})
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig("b2.jpg")


    print(theta)
RLS()