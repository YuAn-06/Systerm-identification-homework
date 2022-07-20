# Copyright (C) 2021 #
# @Time    : 2021/11/16 18:02
# @Author  : Xingyuan Li
# @Email   : 2021200795@buct.edu.cn
# @File    : Msequence.py
# @Software: PyCharm
import numpy as np

def m_sequence(l):
    # 1,0,0,1,1,1
    coef = [1,0,0,1,1,1]
    m = len(coef)


    seq = np.zeros(l)
    #[1,1,0,0,0,1]
    #[0,1,0,0,1,1]
    registers = [1,1,0,1,0,1]
    for i in range(l):
        seq[i] = registers[m - 1]
        C = np.multiply(np.array(coef), np.array(registers))
        C = C.tolist()
        feedback=sum(C) % 2
        registers[1:len(registers)] = registers[0:len(registers) - 1]
        registers[0] = feedback
    return seq