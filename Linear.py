import math
import random
import time

import numpy as np
import  DataLoader
from sklearn import linear_model
import utils
import rule



class Linear:
    def __init__(self,colNum,winSize,confidence,n_components,MaxLen,sampleT = 1.0,pc=1.0,RidgeAll=False,ImplicationCheck = False):
        self.winSize = winSize
        self.colNum = colNum
        self.confidence = confidence
        self.n_cp = n_components
        self.MaxL  = MaxLen
        self.nameList = []
        self.genName()
        self.rules = []
        self.ImpChoice = ImplicationCheck
        self.successImp=0
        self.AllImp=0
        self.sampleTrain =sampleT
        self.pc = pc
        self.RIDGEALL= RidgeAll
        self.impTime = 0
    def genOneHot(self,VecLen):
        X = np.zeros((VecLen),dtype=int)
        cnt = 0
        upB = int(self.MaxL*random.random())
        if upB ==0:
            upB=1
        while(cnt< upB):
            randomIdx = np.random.randint(low=0,high=VecLen,dtype=int)
            if X[randomIdx]==0:
                X[randomIdx]=1
                cnt+=1
        return X

    def checkImp(self,Newrule):
        """
        Check if this set of rules implies a new rule
        """
        # if self.ImpChoice:
        #     return False

        self.AllImp += 1
        if  len(self.rules) == 0:
            return False
        # print("===============================")
        # print("开始检测")
        # print("原约束集合：")
        # for rule0 in self.rules:
        #     rule0.demo()
        # print("新来的")
        # Newrule.demo()
        # print("===============================")
        # tx = time.time()
        if utils.checkIfLinearImp(self.rules,Newrule) == True :

            self.successImp += 1
            return  True
        else:
            # ty = time.time()
            # print("ONCE:", ty - tx)
            return False


    def genName(self):
       for r in range(self.winSize):
           for c in range(self.colNum):
               self.nameList.append("<"+str(r)+"-"+str(c)+">")
    def miniTrainSH(self,X,Y,xnames,yname):
        size = len(xnames)
        name2idx = {}
        idx2name = {}
        for i in range(len(xnames)):
            name2idx[xnames[i]] = i
            idx2name[i] = xnames[i]
        PossibleX = set()
        s2L = {}
        for i in range(5000):
            oneHot = self.genOneHot(size)
            if not 1 in oneHot:
                continue
            PossibleX.add(oneHot.__str__())
            s2L[oneHot.__str__()]= oneHot
        lenI = X.shape[0]
        n = len(PossibleX)
        # print(int(np.log2(n)))
        for k in range(int(np.log2(n))):
            # print(k)
            n = len(PossibleX)
            Loss = {}
            budget = int( lenI / ( n * np.log2(n) ) ) + 10
            trainX = X[:budget,:]
            tranY = Y[:budget]
            strxList = {}
            for X0 in PossibleX:
                xl = s2L[X0]
                # print(X0)
                selectTrainX = 0
                XList = []
                acc = 0
                for idx in range(len(xl)):
                    # print(idx)
                    if xl[idx] == 1:
                        # print("F")
                        # print(len(oneHot))
                        XList.append(xnames[idx])
                        if acc == 0:
                            acc += 1
                            selectTrainX = trainX[:, idx]
                        else:
                            selectTrainX = np.c_[selectTrainX, trainX[:, idx]]
                model = linear_model.Ridge(self.pc)
                # print(selectTrainX )
                # print(selectTrainX.shape)
                if len(selectTrainX.shape) == 1:
                    selectTrainX = selectTrainX.reshape(-1, 1)

                model.fit(selectTrainX, tranY)
                evalResult = np.sum((tranY - model.predict(selectTrainX)) ** 2)
                Loss[X0.__str__()] = evalResult
                strxList[X0.__str__()] = X0
            # print(Loss)
            strValue = sorted(Loss.items(), key=lambda x: x[1], reverse=False)
            PossibleX = set()
            for kv in strValue:
                PossibleX.add(strxList[kv[0]])
                if len(PossibleX) >= n/2:
                    break
            # print(len(PossibleX))
            if(len(PossibleX) <= self.n_cp):
                break
        # print(PossibleX)
        pure = []
        for x in PossibleX:
            pure.append( s2L[x] )
            if len(pure) >= self.n_cp:
                break
        # print("55")
        for xList in pure:
            indexL = []
            xnameList = []
            for x0 in range(len(xList)):
                if xList[x0] == 1:
                    indexL.append(x0)
                    xnameList.append(idx2name[x0])
            r,c = X.shape
            X0 = X[:int(r*self.sampleTrain),indexL]
            y0 = Y[:int(r*self.sampleTrain)]
            # print(X0.shape)
            model = linear_model.Ridge(self.pc)
            # print("FITTING")
            model.fit(X0, y0)
            # print("Done")
            func = {}
            func["Type"]="Linear"
            coefL = []
            for mc in  model.coef_:
                coefL.append(round(mc,2))
            func["coef"] = coefL
            func["intercept"] = round(model.intercept_,2)
            nx = np.array([])
            Loss =( abs(model.predict(X0)-y0) )
            meanLoss = np.mean(Loss)
            m = X0.shape[0]
            b = max(Loss)
            gamma = meanLoss +b*math.sqrt(math.log(1/(1-self.confidence))/(2*m)) + b * math.sqrt(  2* (len(xnames)+1) *( (math.log((math.e*m)/(2* (len(xnames)+1)) ))/m   ) )
            r0 = rule.rule(xnameList,yname,func,-gamma,gamma,eval=model)
            # print("OO")
            if self.ImpChoice == False:
                self.rules.append(r0)
                continue
            else:
                # print("lllll")
                # self.rules.append(r0)
                t0 = time.time()
                result = self.checkImp(r0)
                t1 = time.time()
                deltaT = t1 - t0
                # print("<IMP>")
                # print("Result",result)
                # print(deltaT,self.impTime)
                self.impTime += deltaT
                if result == False:
                    self.rules.append(r0)





    def Train(self,Data,demo=False):

        Data = Data[:,:self.colNum]
        transData = utils.xWindowize(Data,windowsize=self.winSize)
        for i in range(self.winSize*self.colNum):
            print("\rTraing Linear Pct:",i ,"/",(self.winSize * self.colNum),i / (self.winSize * self.colNum), end=' ',flush=True)
            objective = i
            idx=0
            Xnamelist = []
            yname=""
            for r in range(self.winSize):
                for c in range(self.colNum):
                    if idx == objective:
                        yname = "<"+str(r) + "-" + str(c)+">"
                    else:
                        Xnamelist.append("<"+str(r) + "-" + str(c)+">")
                    idx += 1

            Y = transData[:,objective]
            X=0
            if objective == 0:
                X = transData[:,objective+1:]
            else:
                X = np.c_[transData[:,0:objective],transData[:,objective+1:]]
            self.miniTrainSH(X,Y,xnames=Xnamelist,yname=yname)
            # print("done")
        print("\rTraing LinearPct:", 1.0, end='\n')
        if demo:
            print("Done!")
            if self.ImpChoice == True:
                print("Rules Discovered:",self.AllImp)
                print("Rules Implied:", self.successImp)

                print("Proportion of Redundancy Eliminated:",self.successImp,"/",self.AllImp," =",self.successImp/(self.AllImp))
            print("============================")
            print("Rule Discovered:")
            for r in self.rules:
                r.demo()



if __name__ =="__main__":
    DL = DataLoader.DataLoader("NASDAQ")
    x = DL.file2Numpy()
    L = Linear(4,winSize=2,confidence=0.95,n_components=2,MaxLen=3)
    # L.randomOneHot(20)
    L.Train(x,demo=True)