# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 12:36:37 2021

@author: Leonardo
"""


import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--mode", type= str, required= True, help= "Set as train or test")

args = parser.parse_args()

a = np.load(f"linear_rl_trader_rewards/{args.mode}.npy")

print(f"Average Reward: {a.mean():.2f}, Min: {a.min():.2f}, Max:{a.max():.2f}")

plt.hist(a, bins= 20)
plt.title(args.mode)
plt.show()