# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:59:27 2020

@author: Andrew

Reads numerical data from csv and convolves EIIP values of each protein and peptide to output lists of convolution values
intended to be training/testing data for the neural network model.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

rawdata = pd.read_csv("CNN_with_EIIP.csv")


count = 0
for seq in rawdata["conv_length"]:
    x = ast.literal_eval(rawdata.loc[count, "peptide_EIIP"])
    y = ast.literal_eval(rawdata.loc[count, "protein_EIIP"])
    rawdata.loc[count, "conv_length"] = str(np.convolve(x, y, mode = 'same').tolist())
    count= count + 1

rawdata.to_csv('CNN_with_EIIP.csv', index = False)
