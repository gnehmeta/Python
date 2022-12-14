#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
import jacobiPol as jp
#****************************************************************************************
    


#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************    
def orthopoly1D_deriv(x = None,n = None): 
    p = np.zeros((n + 1,1))
    p = np.matrix(p)
    dp = np.zeros((n + 1,1))
    dp = np.matrix(dp)
    p[0,:] = (1 / np.sqrt(2))*np.matrix(np.ones((1,p.shape[1])))
    dp[0,:] = 0*np.matrix(np.ones((1,dp.shape[1])))
    for i in range(1,n+1):
        factor = np.sqrt((2*i + 1) / 2)
        p[i,:] = jp.jacobiP(x,0,0,i) * factor
        dp[i,:] = jp.jacobiP(x,1,1,i - 1) * ((i + 1) / 2) * factor
    return p,dp
#****************************************************************************************    