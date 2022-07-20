# Copyright (C) 2021 #
# @Time    : 2021/11/14 21:20
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : Msequence2.py
# @Software: PyCharm
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


def white_noise(L,mu,sigma):
    def m_sequence(L,A):
        M = 1024
        x0 = 1
        fake_seq = np.zeros(L)
        for i in range(L):
            x = (A * x0) % (M)
            temp = x/M
            fake_seq[i] = temp
            x0 = x
        return fake_seq
    def plot(normal,mu,sigma):
        font = {"size":20}
        plt.figure(2,figsize=(8,6))
        plt.title('White Noise',fontdict=font)
        plt.xlabel('Times ',fontdict={"size":18})
        plt.plot(normal)
        plt.tick_params(labelsize=15)
        plt.legend(['Normal White Noise. ($\mu: ${:.1f} and $\sigma^2: ${:.4f} )'.format(mu, sigma)])
        plt.figure(1)
        sns.distplot(normal, fit=norm, color='red',)
        plt.ylabel('Frequency',fontdict=font)
        plt.legend(['White noise Distribution',
                    'Normal Contribution. ($\mu: ${:.1f} and $\sigma^2: ${:.4f} )'.format(mu, sigma)])
        plt.title('Distribution',fontdict=font)
        plt.grid()
        plt.show()
    seq_one = m_sequence(L,7)
    seq_two = m_sequence(L,3)
    # plt.figure(4,figsize=(8,6))
    # plt.plot(seq_two)
    normal = mu + np.sqrt(sigma)*(np.sqrt(-2*np.log(seq_one)))*np.cos(2*pi*seq_two)
    # plot(normal,mu,sigma)
    # normal = pd.DataFrame(normal,columns='Normal Noise')
    # normal.index = normal.index+1
    # normal.to_excel('White Noise.xls')
    return normal
