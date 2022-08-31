import math
import time

import numpy as np
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicTransformer

import  DataLoader
from sklearn import linear_model
import utils
import rule


def _mape(y, y_pred, w):
    """Calculate the MSE."""
    diffs = (y_pred - y) ** 2
    return 100. * np.average(diffs, weights=w)

class NonLinear:
    def  __init__(self, colNum, winSize, confidence, n_components, MaxLen=4,population = 5000,fitness ="MLE",sampleT = 0.01,pc=0.05,ImplicationCheck=False,generations = 10):
        self.winSize = winSize
        self.colNum = colNum
        self.confidence = confidence
        self.n_cp = n_components
        self.MaxL = MaxLen
        self.generations = generations
        self.fittness = fitness
        self.pop = population
        self.nameList = []
        self.genName()
        self.rules = []
        self.startingPoint = 0
        self.successImp=0
        self.AllImp=0
        self.sampleTrain = sampleT
        self.pc = pc
        self.ImpChoice = ImplicationCheck
        self.impTime = 0
    def genName(self):
       for r in range(self.winSize):
           for c in range(self.colNum):
               self.nameList.append("<"+str(r)+"-"+str(c)+">")



    def genOneHot(self,VecLen):
        X = np.zeros((VecLen),dtype=int)
        cnt = 0
        while(cnt< self.MaxL):
            randomIdx = np.random.randint(low=0,high=VecLen,dtype=int)
            if X[randomIdx]==0:
                X[randomIdx]=1
                cnt+=1
        return X
    def miniTrain(self,X,Y,xnames,yname,demo = False):
        size = len(xnames)
        # name2idx = {}
        # for i in range(len(xnames)):
            # name2idx[xnames[i]] = i
        r,c = X.shape
        sampleSz = int(self.sampleTrain*r)
        trainX = X[:sampleSz,:]
        trainY = Y[:sampleSz]
        mape = make_fitness(function= _mape,
                            greater_is_better=False)
        expression = []
        vb=0
        if demo == True :
            vb  = 1
        ST = SymbolicTransformer(n_components=self.n_cp, function_set=['add', 'sub', 'mul', 'div']
                                 , population_size=self.pop,
                                 n_jobs=12,
                                 init_depth=(1, self.MaxL), feature_names=xnames,
                                 generations=self.generations, stopping_criteria=0.01,
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.9, verbose=vb,
                                 parsimony_coefficient=self.pc, metric=mape,random_state=45)
        ST.fit(X, Y)
        for str0 in ST:
            print(str0)
            func = {}
            func['param'] = str0.__str__()
            Loss = (abs(str0.execute(X) - Y))
            meanLoss = np.mean(Loss)
            m = X.shape[0]
            b = max(Loss)
            gamma = meanLoss + b * math.sqrt(math.log(1 / (1 - self.confidence)) / (2 * m)) + b * math.sqrt(
                2 * (self.MaxL) * ((math.log((math.e * m) / (2 * (self.MaxL)))) / m))
            gammaConfi = b * math.sqrt(math.log(1 / self.confidence) / (2 * m))
            if gamma <= 1e-7:
                gamma = 1e-7

            newX = []
            for x0 in xnames:
                if x0 in str0.__str__():
                    newX.append(x0)
            r0 = rule.rule(newX, yname, func, -gamma, gamma,eval =None )
            x0 = X[0, :]
            y0 = Y[0]
            if len(self.rules) == 0:
                self.rules.append(r0)
                continue

            if self.ImpChoice == False:
                self.rules.append(r0)
                continue
            t0 = time.time()
            result = self.checkImp(r0, x0, y0)
            t1 = time.time()
            deltaT = t1 - t0
            self.impTime += deltaT

            if result:
                continue
            else:
                # if True:
                flag = 1
                self.rules.append(r0)




    def checkImp(self,Newrule,x0,y0):
        """
                Check if this set of rules implies a new rule
                """
        self.AllImp +=1
        if len(self.rules) == 0:
            return False
        # Newrule.demo()
        # print(self.rules)
        # print(Newrule)
        # print(self.nameList)
        if utils.checkNonLinearImp(self.rules, Newrule,self.nameList,self.startingPoint) == True:
            self.successImp+=1
            # print("===============================")
            # print("原约束集合：")
            # for rule0 in self.rules:
            #     rule0.demo()
            # print("新来的")
            # Newrule.demo()
            # print("===============================")
            return True
        else:
            return False

    def Train(self, Data,demo=False,ColumnNames = None):
        Data = Data[:, :self.colNum]
        transData = utils.xWindowize(Data, windowsize=self.winSize)
        self.startingPoint = transData[0,:]
        np.random.shuffle(transData)

        for i in range(self.winSize*self.colNum):
            print("\rTraing NonLinearPct:",str(i),"/",str((self.winSize*self.colNum)),end=' ',flush=True)
            objective = i
            idx=0
            Xnamelist = []
            if ColumnNames == None:
                yname=""
                for r in range(self.winSize):
                    for c in range(self.colNum):
                        if idx == objective:
                            yname ="<" +str(r) + "-" + str(c)+">"
                        else:
                            Xnamelist.append("<"+str(r) + "-" + str(c)+">")
                        idx += 1
            else:
                self.nameList = []
                yname = ""
                for r in range(self.winSize):
                    for c in range(self.colNum):
                        self.nameList.append("<" +ColumnNames[c] + "_i" + str(r)+">")
                        if idx == objective:
                            yname ="<" +ColumnNames[c] + "_i" + str(r)+">"
                        else:
                            Xnamelist.append("<"+ColumnNames[c] + "_i" + str(r)+">")
                        idx+=1

            Y = transData[:, objective]
            X = 0

            if objective == 0:
                X = transData[:, objective + 1:]
            else:
                X = np.c_[transData[:, 0:objective], transData[:, objective + 1:]]
            self.miniTrain(X, Y, xnames=Xnamelist, yname=yname,demo = demo)
        if demo == True:
            print("\rTraing NonLinearPct:",1.0,end='\n')
            print("Search Complete!")
            print("Done!")
            print("============================")
            if self.ImpChoice == True:
                print("Rules Discovered:",self.AllImp+1)
                print("Rules Implied:", self.successImp)

                print("Proportion of Redundancy Eliminated:",self.successImp,"/",self.AllImp+1," =",self.successImp/(1+self.AllImp))
            print("============================")
            for r in self.rules:
                r.demo()
            print("============================")

if __name__ =="__main__":
    DL = DataLoader.DataLoader("NASDAQ")
    x = DL.file2Numpy()
    L = NonLinear(3,winSize=2,confidence=0.95,n_components=1,MaxLen=2,generations=2)
    L.Train(x,demo=True,ColumnNames=["Date","Open","High"])
