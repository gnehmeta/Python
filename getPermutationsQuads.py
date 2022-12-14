#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import numpy as np
#****************************************************************************************



#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************    
def getPermutationsQuads(nDeg = None): 
    if 1 == nDeg:
        perm = np.array([1,2,4,3])
    else:
        if 2 == nDeg:
            perm = np.array([1,5,2,8,9,6,4,7,3])
        else:
            if 3 == nDeg:
                perm = np.array([1,5,6,2,12,13,14,7,11,15,16,8,4,10,9,3])
            else:
                if 4 == nDeg:
                    perm = np.array([1,5,6,7,2,16,17,18,19,8,15,20,21,22,9,14,23,24,25,10,4,13,12,11,3])
                else:
                    if 5 == nDeg:
                        perm = np.array([1,5,6,7,8,2,20,21,22,23,24,9,19,25,26,27,28,10,18,29,30,31,32,11,17,33,34,35,36,12,4,16,15,14,13,3])
                    else:
                        if 6 == nDeg:
                            perm = np.array([1,5,6,7,8,9,2,24,25,26,27,28,29,10,23,30,31,32,33,34,11,22,35,36,37,38,39,12,21,40,41,42,43,44,13,20,45,46,47,48,49,14,4,19,18,17,16,15,3])
                        else:
                            if 7 == nDeg:
                                perm = np.array([1,5,6,7,8,9,10,2,28,29,30,31,32,33,34,11,27,35,36,37,38,39,40,12,26,41,42,43,44,45,46,13,25,47,48,49,50,51,52,14,24,53,54,55,56,57,58,15,23,59,60,61,62,63,64,16,4,22,21,20,19,18,17,3])
                            else:
                                if 8 == nDeg:
                                    perm = np.array([1,5,6,7,8,9,10,11,2,32,33,34,35,36,37,38,39,12,31,40,41,42,43,44,45,46,13,30,47,48,49,50,51,52,53,14,29,54,55,56,57,58,59,60,15,28,61,62,63,64,65,66,67,16,27,68,69,70,71,72,73,74,17,26,75,76,77,78,79,80,81,18,4,25,24,23,22,21,20,19,3])
                                else:
                                    raise Exception('Not implemented yet')
    return perm
#****************************************************************************************    