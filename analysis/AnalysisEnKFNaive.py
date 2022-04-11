# -*- coding: utf-8 -*-

import numpy as np
from analysis.analysis import Analysis

class AnalysisEnKFNaive(Analysis):
  def __init__(self):
    pass;
  
  def performassimilation(self, Xb, H, R, y):
    n, ensemble_size = Xb.shape; 
    Ys = np.random.multivariate_normal(y, R, size=ensemble_size).T;
    D = Ys-H@Xb;
    xb = np.mean(Xb,axis=1);
    DX = Xb-np.outer(xb,np.ones(ensemble_size));
    DXG = 1/(np.sqrt(ensemble_size-1))*DX;
    Q = H@DXG;
    IN = R + Q@Q.T;
    Z = np.linalg.solve(IN,D);
    self.Xa = Xb + DXG@(Q.T@Z);
    return self.Xa;
  
  def getanalysisstate(self):
    return np.mean(self.Xa,axis=1);