import math

import numpy as np
import random
import matplotlib.pyplot as plt
import DataLoader
import utils
from rule import rule


def sub(a, b):
    return a - b


def div(a, b):
    return a / b


def mul(a, b):
    return a * b
def log(a):
    return math.log(a)

def add(a, b):
    return a + b
def sqrt(a):
    return math.sqrt(a)

def AnnomyT(D,corruptRatio ,SDC, ConseqLen = 2 ):
    """
    产生异常
    :param D: 多维时序数据
    :param corruptRatio:  污染的比例
    :param SDC: SpikeError / Consecutive Error
    :return:
    """
    random.seed(42)
    m,n = D.shape
    MaxCol=[]
    for i in range(n):
        MaxCol.append(max(D[:,i]))
    CorruptD =  D.copy()
    Label = np.zeros(m)
    for i in range(m - ConseqLen):
        randomFlag = random.random() < corruptRatio
        if randomFlag:
            Label[i] =1
            colidx = random.randint(0,n-1)
            spikeFlag = random.random() < SDC
            if spikeFlag :
                #SpikeError
                CorruptD[i,colidx] =  MaxCol[colidx] * 0.5
            else:
                for j in range(ConseqLen):
                    Label[i+j] = 1
                    CorruptD[i+j,colidx] = D[0+j,colidx]
                i+=ConseqLen
    return CorruptD,Label

def CORDBasedCheck(ruleSet,D,trueLabel,winSize):
    m,n = D.shape
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(m-winSize+1):
        X = D[i:i+winSize,:]
        realMistake = False
        if 1 in trueLabel[i:i+winSize]:
            realMistake = True
        detect =False
        for r in ruleSet:
            result = eval(r.function)
            if result <= r.ub and result >= r.lb:
                continue
            else:
                detect =True
                break
        if realMistake == True :
            if detect == True:
                TP+=1
            else:
                FN+=1
        else:
            if detect == True:
                FP+=1
            else:
                TN+=1
    print("TP,TN,FP,FN")
    print(TP,TN,FP,FN)
    return TP,TN,FP,FN

def NASDAQExp(cR=0.15):
    DCRuleSet = []
    #从卡塔尔金标准找的否定约束
    #0      1      2    3       4
    #Open, High, Low, Close, AdjClose
    DL = DataLoader.DataLoader("NASDAQ")
    X = DL.file2Numpy()[:,1:6]
    His={}


    X0, Label = AnnomyT(X, corruptRatio=cR, SDC=0.5, ConseqLen=5)
    # Open - High <=0
    rule0 = rule(None, None, "sub(X[0,0],X[0,1])", -1000000, 0, None)
    # Open - Low >=0
    rule1 = rule(None, None, "sub(X[0,0],X[0,2])", 0, 1000000, None)
    # Close - High >=0
    rule2 = rule(None, None, "sub(X[0,3],X[0,1])", -1000000, 0, None)
    # Close - Low  >=0
    rule3 = rule(None, None, "sub(X[0,3],X[0,2])", 0, 1000000, None)
    # AdjClose = Close
    rule4 = rule(None,None,  "sub(X[0,3],X[0,4])",-0.10,0.10 ,None)

    # Open - High <=0
    rule10 = rule(None, None, "sub(X[1,0],X[1,1])", -1000000, 0, None)
    # Open - Low >=0
    rule11 = rule(None, None, "sub(X[1,0],X[1,2])", 0, 1000000, None)
    # Close - High >=0
    rule12 = rule(None, None, "sub(X[1,3],X[1,1])", -1000000, 0, None)
    # Close - Low  >=0
    rule13 = rule(None, None, "sub(X[1,3],X[1,2])", 0, 1000000, None)
    # AdjClose = Close
    rule14 = rule(None, None, "sub(X[1,3],X[1,4])", -0.10, 0.10, None)



    ruleSet = [rule4,rule3,rule2,rule0,rule1,rule11,rule12,rule13,rule14]
    TP,TN,FP,FN=CORDBasedCheck(ruleSet, X0, Label, 2)
    AccDC = ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)

    print(AccDC,F1)
    print("++++++++++++++++")
    His["DC-Acc"] = AccDC
    His["DC-F1"] = F1
    #RowwiseCORD
    # sub(add(X[0,1], -0.682),X[0,0])

    #
    rule0 = rule(None, None, "sub( X[0,0] ,X[0,1])", -160,  160, None)
    # Open - Low >=0
    rule1 = rule(None, None, "sub(X[0,0],X[0,2])", -160, 160, None)
    # Close - High >=0
    rule2 = rule(None, None, "sub(X[0,3],X[0,1])", -100, 100, None)
    # Close - Low  >=0
    rule3 = rule(None, None, "sub(X[0,3],X[0,2])",  -160, 160, None)
    # AdjClose = Close
    rule4 = rule(None, None, "sub(X[0,3],X[0,4])", -0.01, 0.01, None)

    rule10 = rule(None, None, "sub( X[1,0] ,X[1,1])", -160, 160, None)
    # Open - Low >=0
    rule11 = rule(None, None, "sub(X[1,0],X[1,2])", -160, 160, None)
    # Close - High >=0
    rule12 = rule(None, None, "sub(X[1,3],X[1,1])", -100, 100, None)
    # Close - Low  >=0
    rule13 = rule(None, None, "sub(X[1,3],X[1,2])",  -160, 160, None)
    # AdjClose = Close
    rule14 = rule(None, None, "sub(X[1,3],X[1,4])", -0.001, 0.001, None)
    rule21 = rule(None, None, "sub(X[0,0],X[1,0])", -63.282, 63.282, None)
    rule22 = rule(None, None, "sub(X[0,1],X[1,1])", -80, 80, None)
    rule23 = rule(None, None, "sub(X[0,2],X[1,2])", -73.27, 73.27, None)
    rule24 = rule(None, None, "sub(X[0,3],X[1,3])", -83.27, 83.27, None)
    rule25 = rule(None, None, "sub(X[0,4],X[1,4])", -83.27, 83.27, None)

    ruleSet = [rule4, rule3, rule2, rule0, rule1,rule11,rule12,rule13,rule14,rule10,rule24, rule23, rule22, rule25, rule21]
    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label, 2)
    AccCORD = ((TP + TN) / (TP + FP + TN + FN))
    # print(AccCORD)
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)

    print(AccCORD, F1)
    print("++++++++++++++++")
    His["CORD-Acc"] = AccCORD
    His["CORD-F1"] = F1
    rule21 = rule(None, None, "sub(X[0,0],X[1,0])", -64, 64, None)
    rule22 = rule(None, None, "sub(X[0,1],X[1,1])", -80, 80, None)
    rule23 = rule(None, None, "sub(X[0,2],X[1,2])", -73.27, 73.27, None)
    rule24 = rule(None, None, "sub(X[0,3],X[1,3])", -83.27, 83.27, None)
    rule25 = rule(None, None, "sub(X[0,4],X[1,4])", -83.27, 83.27, None)
    ruleSet = [rule24, rule23, rule22, rule25, rule21]
    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label,2)
    AccSD = ((TP + TN) / (TP + FP + TN + FN))


    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)

    print(AccSD, F1)
    print("++++++++++++++++")
    His["SD-Acc"] = AccSD
    His["SD-F1"] = F1
    print(AccCORD, AccDC , AccSD )
    return His
def WindExp(CR=0.1,mode="A"):
    # 引风机3A总有功功率,引风机3A出口烟气压力,引风机3A电流,引风机3A动叶调节挡板阀位,引风机3A驱动端轴承温度1,引风机3A非驱动端轴承温度2,
    DL = DataLoader.DataLoader("WindA")
    if mode == "A":
        X = DL.file2Numpy()[:10000,:]
    else:
        X = DL.file2Numpy()[10000:20000,:]
    His ={}
    print(X.shape)
    # r,c = X.shape
    # for i in range(c):
    #     print(max(X[:,i]),min(X[:,i]))
    # exit(1)
    X0, Label = AnnomyT(X, corruptRatio=CR, SDC=0.5, ConseqLen=5)
    #一些 SD
    # 引风机3A总有功功率[i]  <-  ['引风机3A总有功功率[i+1]']
    rule0 = rule(None, None, "sub(X[0,0],X[1,0])", -511.1, 512.070, None)
    #  引风机3A出口烟气压力[i]  <-  ['引风机3A出口烟气压力[i+1]']
    rule1 = rule(None, None, "sub(X[0,1],X[1,1])", -0.6, 0.6, None)
    # 引风机3A电流[i]  <-  ['引风机3A电流[i+1]']
    rule2 = rule(None, None, "sub(X[0,2],X[1,2])", -32, 32, None)
    # 引风机3A动叶调节挡板阀位[i]  <-  ['引风机3A动叶调节挡板阀位[i+1]']
    rule3 = rule(None, None, "sub(X[0,3],X[1,3])", -10.3, 10.3, None)
    # 引风机3A驱动端轴承温度1[i]  <-  ['引风机3A驱动端轴承温度1[i+1]']
    rule4 = rule(None,None,  "sub(X[0,4],X[1,4])",-9.3,9.3 ,None)
    # 引风机3A非驱动端轴承温度2[i]  <-  ['引风机3A非驱动端轴承温度2[i+1]']
    rule5 = rule(None, None, "sub(X[0,5],X[1,5])", -12.6, 12.6, None)

    ruleSet = [rule0,rule1,rule2,rule3,rule4,rule5]
    TP,TN,FP,FN=CORDBasedCheck(ruleSet, X0, Label, 2)
    AccSD= ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)
    print("SD")
    print(AccSD,F1)
    His["SD-Acc"] = AccSD
    His["SD-F1"] = F1
    # 引风机3A总有功功率[i]  <-  ['引风机3A总有功功率[i+1]']
    rule0 = rule(None, None, "sub(X[0,0],X[1,0])", -511.1, 512.070, None)
    #  引风机3A出口烟气压力[i]  <-  ['引风机3A出口烟气压力[i+1]']
    rule1 = rule(None, None, "sub(X[0,1],X[1,1])", -0.6, 0.6, None)
    # 引风机3A电流[i]  <-  ['引风机3A电流[i+1]']
    rule2 = rule(None, None, "sub(X[0,2],X[1,2])", -32, 32, None)
    # 引风机3A动叶调节挡板阀位[i]  <-  ['引风机3A动叶调节挡板阀位[i+1]']
    rule3 = rule(None, None, "sub(X[0,3],X[1,3])", -10.3, 10.3, None)
    # 引风机3A驱动端轴承温度1[i]  <-  ['引风机3A驱动端轴承温度1[i+1]']
    rule4 = rule(None,None,  "sub(X[0,4],X[1,4])",-9.3,9.3 ,None)
    # 引风机3A非驱动端轴承温度2[i]  <-  ['引风机3A非驱动端轴承温度2[i+1]']
    rule5 = rule(None, None, "sub(X[0,5],X[1,5])", -12.6, 12.6, None)
   # 引风机3A总有功功率[i]  <-  ['引风机3A动叶调节挡板阀位[i]', '引风机3A驱动端轴承温度1[i]']
    rule6 = rule(None, None, "sub(X[0,0],mul(add(X[0,3],X[0,3]),X[0,4]))", -932.62, 932.6, None)

    rule7 = rule(None, None, "sub(X[1,0],mul(add(X[1,3],X[1,3]),X[1,4]))", -932.62, 932.6, None)

    rule8 = rule(None,None, "sub(log(X[0,5]),X[0,1])",-2,2,None)


    rule9 = rule(None,None,"sub(X[0,2],mul(X[0,1],X[0,3]))",-280,280,None)

    rule10 = rule(None, None, "sub(X[1,2],mul(X[1,1],X[1,3]))", -280, 280, None)

    rule11 = rule(None,None,"sub(X[0,4],X[0,5])",-14,14,None)

    rule12 = rule(None,None,"sub(X[1,4],X[1,5])",-14,14,None)

    ruleSet = [rule0,rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12]


    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label, 2)
    AccSD = ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)
    print("CORD")
    print(AccSD,F1)
    His["CORD-Acc"] = AccSD
    His["CORD-F1"] = F1
    # DC
    rule0x = rule(None, None, "sub(X[0,0],0)", 4012, 5573, None)
    rule0x1 = rule(None, None, "sub(X[0,1],0)", 0, 5, None)

    rule0x2 = rule(None, None, "sub(X[0,2],0)", 261, 350, None)
    rule0x3 = rule(None, None, "sub(X[0,3],0)",39, 55, None)
    rule0x4 = rule(None, None, "sub(X[0,4],0)", 40, 60, None)
    rule0x5 = rule(None, None, "sub(X[0,5],0)", 40, 60, None)

    # rule0y = rule(None, None, "sub(X[0,1],3.69)", 0, 3.9, None)
    rule00 = rule(None,None,  "sub(X[0,0],X[0,1])",0,100000,None)
    rule01 = rule(None, None, "sub(X[0,1],X[0,2])", -100000, 0, None)
    rule02 = rule(None, None, "sub(X[0,0],X[0,3])", 0, 10000, None)
    rule03 = rule(None, None, "sub(X[0,0],X[0,4])", 0, 10000, None)
    rule04 = rule(None, None, "sub(X[0,0],X[0,5])", 0, 10000, None)
    ruleSet = [rule00,rule01,rule02,rule03,rule04,rule0x,rule0x1,rule0x2,rule0x3,rule0x4,rule0x5]
    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label, 1)
    AccDC = ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)
    print("DC")
    print(AccDC, F1)
    His["DC-Acc"] = AccDC
    His["DC-F1"] = F1

    return His

def BoatExp(corruptR = 0.2):
    DL = DataLoader.DataLoader("Boat")
    X = DL.file2Numpy()
    X0, Label = AnnomyT(X, corruptRatio=corruptR, SDC=1, ConseqLen=5)
    His={}
    # print(X[:5,:])
    # exit(1)
    #CORD
    rule00 = rule(None, None, "sub(X[0,0],0)", -5, 0, None)
    rule01 = rule(None, None, "sub(X[0,1],0)", 0.52, 0.6, None)
    rule02 = rule(None, None, "sub(X[0,2],0)", 4.3, 5.2, None)
    rule03 = rule(None, None, "add(X[0,3],0)", 2.8, 5.35 , None)
    rule04 = rule(None, None, "sub(X[0,4],0)", 2.7, 3.64, None)
    rule05 = rule(None, None, "sub(X[0,5],0)", 0.12, 0.46, None)
    rule06 = rule(None, None, "sub(X[0,6],0)", 0, 63, None)

    rule07 = rule(None, None, "sub(X[0,0], mul(X[0,3],-0.768) )",  -3.6  , 3.6 , None)
    rule08 = rule(None,None,"sub(X[0,1], div(X[0,2],add(X[0,4],X[0,2])) )",-0.08,0.08,None)
    rule09 = rule(None, None, "sub(X[0,2], add(X[0,3],0.684) )", -1.1, 1.1, None)
    rule10 = rule(None,None,"sub(X[0,3], add( mul(X[0,2],X[0,1]) , mul(X[0,2],X[0,5]) ) )",-1.3,1.3,None)

    rule11 = rule(None,None,"sub(X[0,6], add(X[0,2],mul(X[0,1],X[0,2])))",-52,52,None)
    rule12 = rule(None, None, "sub(X[0,5], div(0.932,X[0,4]) )", -0.3, 0.3, None)
    rule13 = rule(None, None, "sub(X[0,5], mul(0.569,X[0,1]) )", -0.25, 0.25, None)
    rule111 = rule(None,None,"sub(X[0,6], add(mul( div(1,mul(X[0,5],X[0,5]) ) , -0.47 )  ,19.53) )",-38.8,38.8,None)
    ruleSet = [rule00 , rule02 ,rule03 ,rule04,rule05,rule06,rule07 ,rule08 ,rule111,rule12,rule13]
    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label, 1)
    Acc = ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)
    # print("CORD")
    # print(Acc, F1)
    His["CORD-Acc"] = Acc
    His["CORD-F1"] = F1

    # print("DC")
    #DC
    ruleDC1 =  rule(None, None, "sub(X[0,0],X[0,1])", -99999, 0, None)
    ruleDC2 = rule(None, None, "sub(X[0,2],X[0,4])", 0, 9999990, None)
    ruleDC3 = rule(None, None, "sub(X[0,1],X[0,5])", 0, 9999990, None)

    ruleSet = [   rule02 ,rule03 ,rule04,rule05,rule06,ruleDC1,ruleDC2,ruleDC3]
    TP, TN, FP, FN = CORDBasedCheck(ruleSet, X0, Label, 1)
    Acc = ((TP + TN) / (TP + FP + TN + FN))
    pre = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = 2 * pre * Recall / (pre + Recall)
    print(Acc, F1)
    His["DC-Acc"] = Acc
    His["DC-F1"] = F1
    return His

def BigNASDAQ():
    CL=[0.1,0.2,0.3,0.4]
    head = ["Method","NoiseRate","Acc","F1"]
    Contents = []
    MethList = ["CORD","DC","SD"]
    for cr in CL:
        His = NASDAQExp(cr)
        for meth in MethList:
            row = []
            row.append(meth)
            row.append(cr)
            row.append(His[meth+"-Acc"])
            row.append(His[meth + "-F1"])
            Contents.append(row)
    utils.write2CSV("NASDAQ_OutlierDetection.csv",head,Contents)
def BigBoat():
    CL=[0.1,0.2,0.3,0.4]
    head = ["Method","NoiseRate","Acc","F1"]
    Contents = []
    MethList = ["CORD","DC"]
    for cr in CL:
        His = BoatExp(cr)
        for meth in MethList:
            row = []
            row.append(meth)
            row.append(cr)
            row.append(His[meth+"-Acc"])
            row.append(His[meth + "-F1"])
            Contents.append(row)
    utils.write2CSV("Boat_OutlierDetection.csv",head,Contents)

def BigWind():
    CL = [0.1, 0.2, 0.3, 0.4]
    head = ["Method", "NoiseRate", "Acc", "F1"]
    Contents = []
    MethList = ["CORD", "DC", "SD"]
    for cr in CL:
        His = WindExp(cr,"A")
        for meth in MethList:
            row = []
            row.append(meth)
            row.append(cr)
            row.append(His[meth + "-Acc"])
            row.append(His[meth + "-F1"])
            Contents.append(row)

    utils.write2CSV("WindA_OutlierDetection.csv", head, Contents)
    Contents = []
    print("A-Done")
    for cr in CL:
        His = WindExp(cr, "B")
        for meth in MethList:
            row = []
            row.append(meth)
            row.append(cr)
            row.append(His[meth + "-Acc"])
            row.append(His[meth + "-F1"])
            Contents.append(row)
    utils.write2CSV("WindB_OutlierDetection.csv", head, Contents)


if __name__ == "__main__":
    BigBoat()
