import numpy as np

from Linear import Linear
from NonLinear import NonLinear

# T = np.linspace(1,2000,2000)# Time is Linearly in creasing
# # The first and second cols have no relationships
# I = np.random.randint(500,50000,2000)
# U = np.random.randint(9000,10000,2000)
# # The third col :According to the laws of physics, Power is equal to current times voltage
# P = U*I
# D = np.zeros((2000,4))
# D[:,0] = T
# D[:,1] = I
# D[:,2] = U
# D[:,3] = P
#
# L = NonLinear(4, winSize=2, confidence=0.95, n_components=1, generations=20,ImplicationCheck=False)
# L.Train(D, demo=True, ColumnNames=["T", "I","U", "P"])

A0 = np.linspace(1,2000,2000)# Linearly in creasing
A1 = np.random.randint(-500,500,2000)
A2 = A0+A1
D = np.zeros((2000,3))
D[:,0] = A0
D[:,1] = A1
D[:,2] = A2
L = Linear(3,winSize=2,confidence=0.95,n_components=1,MaxLen=3,ImplicationCheck =True)
L.Train(D,demo=True)
