import numpy as np
import random
import matplotlib.pyplot as plt
import Linear
import NonLinear
import rule
import utils
from DataLoader import DataLoader
import matplotlib as mpl
from matplotlib import font_manager#导入字体管理模块
# Generate some Data which obeys some certain distribution
# 咱们分几个阶段的实验
#       第一阶段：首先不加噪声，看大于，小于的关系生成的能否挖掘出来，
#       第二阶段：然后看线性以及非线性关系
#       第三阶段：看加噪声以后的表现情况


def Task1(noiseLevel):
    extraColNum = 5
    DL= DataLoader("NASDAQ")
    X = DL.file2Numpy()[:,1]
    print(X)
    D = []
    func = {}
    func["Type"] = "Linear"
    func["coef"] = np.array([1])
    func["intercept"] = 0
    f0 = func
    gr0 = rule.rule(["0-0"],"0-1",f0,-5,99999,None)
    gr1 = rule.rule(["0-0"],"0-2",f0,-5,99999,None)
    gr2 = rule.rule(["0-0"],"0-3",f0,-99999,5,None)
    gr3 = rule.rule(["0-0"],"0-4",f0,-5,99999,None)
    GoldenRuleSet=[gr0,gr1,gr2,gr3]

    for rowAx0 in X:
        r0 = []
        v1 = rowAx0 + random.random()*noiseLevel
        v2 = rowAx0 + random.random() * noiseLevel
        v3 = rowAx0 - random.random() * noiseLevel
        v4 = rowAx0 + random.random() * noiseLevel
        r0.append(rowAx0)
        r0.append(v1)
        r0.append(v2)
        r0.append(v3)
        r0.append(v4)
        D.append(r0)
    D = np.array(D)
    LinearJob = Linear.Linear(5, 1, 0.05, 2, 1)
    LinearJob.Train(D)
    RecallV = 0
    for r in GoldenRuleSet:
        if utils.checkIfLinearImp(ruleSet=LinearJob.rules,newrule=r):
            print("OK")
            RecallV+=1
            # exit(1)
        else:
            print("Not ok")
            r.demo()
            print("==============")
    print("Recall:", RecallV / len(GoldenRuleSet))
    return  RecallV / len(GoldenRuleSet)



def Task2(noiseLevel):
    extraColNum = 5
    DL= DataLoader("NASDAQ")
    X = DL.file2Numpy()[:,1]
    print(X)
    D = []
    func = {}
    func["param"] = "sub(0-0,0)"
    f0 = func
    gr0 = rule.rule(["0-0"],"0-1",f0,-10,99999,None)
    gr1 = rule.rule(["0-0"],"0-2",f0,-10,99999,None)
    gr2 = rule.rule(["0-0"],"0-3",f0,-99999,10,None)
    gr3 = rule.rule(["0-0"],"0-4",f0,-10,99999,None)
    GoldenRuleSet=[gr0,gr1,gr2,gr3]

    for rowAx0 in X:
        r0 = []
        v1 = rowAx0 + random.random()*noiseLevel
        v2 = rowAx0 + random.random() * noiseLevel
        v3 = rowAx0 - random.random() * noiseLevel
        v4 = rowAx0 + random.random() * noiseLevel
        r0.append(rowAx0)
        r0.append(v1)
        r0.append(v2)
        r0.append(v3)
        r0.append(v4)
        D.append(r0)
    D = np.array(D)
    NonLinearJ = NonLinear.NonLinear(5,1,0.01,1,1)
    NonLinearJ.Train(D)
    recall = 0
    # exit(1)
    for r in GoldenRuleSet:
        if utils.checkNonLinearImp(ruleSet=NonLinearJ.rules, newrule=r, namelist=["0-0", "0-1", "0-2", "0-3", "0-4"],startPoint=D[0,:]):
            print("OK")
            recall+=1
    print("Recall:",recall/len(GoldenRuleSet))
    return recall/len(GoldenRuleSet)


def stylePlot(figName,X,Ys,xLabelName,yLabelName,param=None):
    """
    自定义格式的绘图函数
    :param figName: 你需要保存的图片的名字，自动存储在./pic路径下
    :param X: 横坐标
    :param Ys: 纵坐标们，是一个字典 key：曲线名字，value:该曲线纵坐标列表
    :param xLabelName: x轴名字
    :param yLabelName: y轴名字
    """
    plt.rc('font', family='Arial')
    tickSize = 20 #坐标轴刻度大小
    xyLineWidth = 1.5 # xy底部坐标轴的粗细
    legendSize= 21#标识字体大小
    plotlineWidth = 2.5
    markerSize = '12'
    xylabel = tickSize+1 #xy轴字体大小
    xwidth = 11
    ylength = xwidth*0.51
    plt.figure(figsize=(xwidth, ylength), dpi=75)
    ax = plt.gca()
    if not param == None :
        a=1+1
        # ax.set_ylim(12, 29)
    else:
        a=1+1
        # ax.set_ylim(0, 1.1)
    Linetype = ["o-", "x-", "*-", "|-", "-", "o-"]
    # Linetype = ["k--", "b-", "r-", "|-", "-", "o-"]
    # Linetype = [ "x-", "*-", "|-", "o-", "o-"]
    idx = 0
    for Linename in  Ys.keys():
        Y = Ys[Linename]
        plt.plot(X,Y,Linetype[idx],linewidth=plotlineWidth,markersize=markerSize,label=Linename,markerfacecolor='none')
        idx+=1
    # #plt.legend()
    my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/STZHONGS.TTF",size=legendSize)
    labelss = plt.legend(fontsize=legendSize,frameon=False).get_texts()

    for label in labelss:
        label.set_fontproperties(my_font)
    # [label.set_fontname('Arial') for label in labelss]



    ax = plt.gca()  # 获得坐标轴的句柄
    plt.tick_params(labelsize=tickSize)     #刻度大小
    ax.spines['bottom'].set_linewidth(xyLineWidth)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(xyLineWidth)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(xyLineWidth)
    ax.spines['top'].set_linewidth(xyLineWidth)
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # my_font = font_manager.FontProperties(fname="C:/WINDOWS/Fonts/STZHONGS.TTF")
    #
    # plt.xlabel(xLabelName,fontproperties = my_font,fontsize=xylabel)
    # plt.ylabel(yLabelName,fontproperties = my_font,fontsize=xylabel)
    plt.xlabel(xLabelName, fontsize=xylabel)
    plt.ylabel(yLabelName, fontsize=xylabel)
    plt.subplots_adjust(top=0.88,
                        bottom=0.16,
                        left=0.140,
                        right=0.920,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig("./pic/"+figName,format="eps")

    plt.show()


# 实验分四组：1. 行上：只包含线性的结构 2. 行上: 混合 3.包含列上的关系：线性 4. 包含列上关系:混合
# 规则金标准： 20条


def Task3(noiseLevel):
    extraColNum = 5
    DL= DataLoader("NASDAQ")
    X = DL.file2Numpy()[:,1]
    print(X)
    D = []
    func = {}
    func["param"] = "sub(0-0,0)"
    f0 = func
    gr0 = rule.rule(["0-0"],"0-1",f0,-5,99999,None)
    func1 = {}
    func1['param'] = "mul(0-0,0-1)"
    gr1 = rule.rule(["0-0","0-1"],"0-2",func1,-5,99999,None)
    func2 = {}
    func2['param'] = "div(0-0,0-1)"
    gr2 = rule.rule(["0-0","0-1"],"0-3",func2,-5,99999,None)
    func3 = {}
    func3['param'] = "add(div(0-1,0-2),0-3)"
    gr3 = rule.rule(["0-0"],"0-4",func3,-5,99999,None)
    GoldenRuleSet=[gr0,gr1,gr2,gr3]

    for rowAx0 in X:
        r0 = []
        v1 = rowAx0 + random.random()*noiseLevel

        v2 = rowAx0 * v1 + random.random()*noiseLevel
        v3 = rowAx0/v1 + random.random()*noiseLevel

        v4 = v1/v2+v3 + random.random()*noiseLevel

        r0.append(rowAx0)
        r0.append(v1)
        r0.append(v2)
        r0.append(v3)
        r0.append(v4)
        D.append(r0)
    D = np.array(D)
    NonLinearJ = NonLinear.NonLinear(colNum=5,winSize=1,confidence=0.1,n_components=2,MaxLen=3)
    NonLinearJ.Train(D)
    recall = 0
    for r in GoldenRuleSet:
        if utils.checkNonLinearImp(ruleSet=NonLinearJ.rules, newrule=r, namelist=["0-0", "0-1", "0-2", "0-3", "0-4"],startPoint=D[0,:]):
            print("OK")
            recall+=1
    print("Recall:",recall/len(GoldenRuleSet))
    return  recall/len(GoldenRuleSet)

if __name__=="__main__":
    X = np.array([0.1,6,7,8])
    recLine = []
    recNonLine = []
    for noise in X:
        recNonLine.append(Task2(noise))
        recLine.append( Task1(noise))
    stylePlot("Task3",X=X,Ys={"LinearResult":recLine,"NonLinearResult":recNonLine},xLabelName="noiseLevel",yLabelName="G-recall")
