import csv
import math

import numpy as np
import  DataLoader
from scipy.optimize import linprog,minimize,NonlinearConstraint
import rule
from functools import partial
import warnings
warnings.filterwarnings("ignore")
def sub(a, b):
    return a - b


def div(a, b):
    if b == 0:
        return 9999999999
    return a / b


def mul(a, b):
    return a * b


def add(a, b):
    return a + b

def log(a):
    if a <= 0 :
        return 9999
    return math.log(a)

def sqrt(a):
    if a <=0:
        return 0
    return math.sqrt(a)


def xWindowize(X ,windowsize = 1):
    """
    give X,transform it into windowize array
    :param X: numpy array
    :return:
    """
    r,c = X.shape
    newArray = []
    for i in range(r-windowsize+1):
        newArray.append(X[i:i+windowsize,:].reshape(-1))
    newArray = np.array(newArray)
    # print(newArray.shape)
    return  newArray
def checkNonLinearImp(ruleSet,newrule,namelist,startPoint):
    """
    目前没有解决形如：a<b<c<d,(a,c) 与(b,d)的蕴含。这个得原子化规则。暂时没有进行。
    咱们只能够（a,d）->(b,c)
    :param ruleSet:
    :param newrule:
    :param namelist:
    :param startPoint:
    :return:
    """


    newRuleSet = []
    name2idx = {}
    inc = 0
    name2val = {}
    # print(len(namelist))
    # print(len(startPoint))
    for n in range(len(namelist)):
        name2val[ namelist[n] ]= startPoint[n]

    for xname in newrule.Xnamelist:
        if xname in name2idx.keys():
            continue
        else:
            name2idx[xname] = inc
            inc += 1
    # print(name2idx)
    if not newrule.Yname in name2idx.keys():
        name2idx[newrule.Yname] = inc
        inc += 1
    # print(name2idx)
    # exit(1)
    for rule0 in ruleSet:
        for xname in rule0.Xnamelist:
            if xname in name2idx.keys():
                continue
            else:
                name2idx[xname] = inc
                inc+=1
        if not rule0.Yname in name2idx.keys():
            name2idx[rule0.Yname] = inc
            inc+=1

        if rule0.lb == "-inf" or rule0.ub == "+inf":
            newRuleSet.append(rule0)
        else:
            r1,r2 = rule0.normalize()
            newRuleSet.append(r1)
            newRuleSet.append(r2)
    selectStart = np.zeros(len(name2idx.keys()))
    # # print()
    # print(name2idx)
    # print(name2val)
    # exit(1)
    for k in name2idx.keys():
        v = name2idx[k]
        selectStart[v] = name2val[k]

    rule2code={}

    # print(name2idx)
    neqPool =[]
    for r0 in ruleSet:
        fx = "sub( "+ r0.function['param']+" , "+r0.Yname+" )"
        neq1 = "sub( "+ fx + " , "+ str(r0.lb) + " )"
        neq2 = "sub( " + str(r0.ub)+ " , " + fx + " )"
        # print(neq1,neq2)
        for n0 in name2idx.keys():
        
            neq1 = neq1.replace(n0," x["+str(name2idx[n0]) +"] ")
            neq2 = neq2.replace(n0, " x[" + str(name2idx[n0]) + "] ")
        # print(neq1,neq2)
        # exit(1)
        neqPool.append(neq1)
        neqPool.append(neq2)
        # print([i(np.zeros(5)) for i in constrain])
    # exit(1)

    def f_constraint(s, index):
        x =s
        # print(neqPool[index],s,eval(neqPool[index]))
        return eval(neqPool[index])
    cons = []

    for ii in range(len(neqPool)):
        # the value of ii is set in each loop
        cons.append({'type': 'ineq', 'fun': partial(f_constraint, index=ii)})
    # print(newrule.function['param'])
    func = "sub( "+newrule.function['param']  +" , " + newrule.Yname + " ) "
    eq1 = "sub( "+ func + " , " + str(newrule.lb) +" )" # fx-lb >= 0
    eq2 = "sub( "+ str(newrule.ub) + " , "+ func + " )" # ub-fx >= 0
    # print(eq1,eq2)
    # # exit(1)
    # print(name2idx)
    for n0 in name2idx.keys():

        eq1 = eq1.replace(n0, " x[" + str(name2idx[n0]) + "] ")
        eq2 = eq2.replace(n0, " x[" + str(name2idx[n0]) + "] ")
        # print(eq1,eq2)
    f1 = lambda x: eval(eq1)
    #
    # print(eq1,eq2)
    # print(name2idx)
    # exit(1)
    # print(neqPool)
    # print(eq1)
    res1 = minimize(f1, x0=selectStart,   constraints=cons)
    # print(eq2)
    f2 = lambda x: eval(eq2)
    res2 = minimize(f2, x0=selectStart,   constraints=cons)
    x0 = res2["x"]
    # for cons1 in cons:
    #     print(cons1['fun'](x0))
    # print(f2(x0))
    # exit(1)
    # print()
    # print()
    # print(res1['message'],res2['message'])
    # print(res1['success'],res2['success'])
    # print(res1['fun'], res2['fun'])
    if res1['fun'] >=-1e-11and res2['fun'] >=-1e-11 :
        # print("Imp Yes!")
        # print(res1['fun'] ,res2['fun'] )
        return True
    else:

        return False






def checkIfLinearImp(ruleSet,newrule):


    # First transform the ruleSet into its normal form
    # print("规则集：")
    # for r in ruleSet:
    #     r.demo()
    # print("=================")
    # print("新来的")
    # newrule.demo()

    newRuleSet = []
    name2idx = {}
    inc = 0
    for xname in newrule.Xnamelist:
        if xname in name2idx.keys():
            continue
        else:
            name2idx[xname] = inc
            inc += 1
    if not newrule.Yname in name2idx.keys():
        name2idx[newrule.Yname] = inc
        inc += 1

    for rule0 in ruleSet:
        for xname in rule0.Xnamelist:
            if xname in name2idx.keys():
                continue
            else:
                name2idx[xname] = inc
                inc+=1
        if not rule0.Yname in name2idx.keys():
            name2idx[rule0.Yname] = inc
            inc+=1

        if rule0.lb == "-inf" or rule0.ub == "+inf":
            newRuleSet.append(rule0)
        else:
            r1,r2 = rule0.normalize()
            newRuleSet.append(r1)
            newRuleSet.append(r2)
    #
    # for r in newRuleSet:
    #     r.demo()
    # print(name2idx)

    # The normalForm of a python Scipy obj is a Ax <= b
    # Have to do 2 extra things :
    # 1 : move the Y into LeftSide
    # 2 : move the intercept into rightSide
    A = []
    b = []

    for normrule in newRuleSet:
        # normrule.demo()
        zeroList = np.zeros((len(name2idx.keys(),)))
        for i in range(len(normrule.Xnamelist)):
            name = normrule.Xnamelist[i]
            coef = normrule.function['coef'][i]
            if "neg" in normrule.function.keys():
                coef = -coef
            zeroList[ name2idx[name] ] = coef

        yname = normrule.Yname
        coef = -1
        b0 = normrule.function["intercept"]
        if "neg" in normrule.function.keys():
            coef = -coef
            b0 = -b0
        # print(len(zeroList),name2idx[yname])
        zeroList[name2idx[yname]] = coef
        A.append(zeroList)
        b.append(normrule.ub-b0)
    A = np.array(A)
    b = np.array(b)

    # print(A)
    # print(b)
    # print(name2idx)
    # exit(1)
    # print(A)
    # print(b)
    zeroList = np.zeros((len(name2idx.keys(), )))
    for i in range(len(newrule.Xnamelist)):
        # normrule.demo()
        # print(i)
        name = newrule.Xnamelist[i]
        coef = newrule.function['coef'][i]
        zeroList[name2idx[name]] = coef
    zeroList[name2idx[newrule.Yname]] = -1
    # print(zeroList)
    # exit(1)
    options = {}
    try:
        options['cholesky'] = False
        options['sym_pos'] = False
        options['lstsq'] = True
        options['presolve'] = False
        res = linprog(c=zeroList, A_ub=A, b_ub=b, options=options)

        ubres = (res['fun']+newrule.function["intercept"]-newrule.lb)
        res2 = linprog(c=-zeroList, A_ub=A, b_ub=b, options=options)
        lbres = (res2['fun'] - newrule.function["intercept"] + newrule.ub)

    except:
        try:
            options['presolve'] =False
            res = linprog(c=zeroList, A_ub=A, b_ub=b, options=options)

            ubres = (res['fun'] + newrule.function["intercept"] - newrule.lb)
            res2 = linprog(c=-zeroList, A_ub=A, b_ub=b, options=options)
        except:
            return False
    # print(res)
    # print(res2)
        lbres = (res2['fun']-newrule.function["intercept"] + newrule.ub)
    # print(res)
    # print(res2)
    # exit(1)
    if lbres >=-1e-6 and ubres>=-1e-6:
        # print("Imp Yes")
        return True
    else:
        # print(res)
        # print(res2)
        return False
    # ObjSet = newrule.normalize()
    # Clist = []
    # for normrule in ObjSet:
    #     ci = []
    #
    #
    # for normrule in ObjSet:
    #     if normrule.lb=="-inf" and normrule.ub == "+inf":
    #         continue
    #     else:
    #         # start 2 transform problem into a LP problem
    #         zeroList = np.zeros((len(name2idx.keys(), )))
    #         # bounds = []
    #         # for i in range(len(name2idx.keys())):
    #         #     bounds.append([-10000,10000])
    #         for i in range(len(normrule.Xnamelist)):
    #             # normrule.demo()
    #             # print(i)
    #             name = normrule.Xnamelist[i]
    #             coef = normrule.function['coef'][i]
    #             if "neg" in normrule.function.keys():
    #                 coef = -coef
    #             zeroList[name2idx[name]] = coef
    #
    #         yname = normrule.Yname
    #         coef = -1
    #         if "neg" in normrule.function.keys():
    #                 coef = -coef
    #         zeroList[name2idx[yname]] = coef
    #
    #         b0 = normrule.function["intercept"]
    #         options = {}
    #         options['cholesky'] = False
    #         options['sym_pos'] = False
    #         options['lstsq'] = True
    #         options['presolve'] = True
    #         print(-zeroList)
    #         res = linprog(c=zeroList, A_ub=A, b_ub=b, options=options)
    #         Minvalue = res['fun']-b0 + normrule.ub
    #         print(Minvalue)
    #         if Minvalue >=0:
    #             continue
    #
    #
    #         # if "neg" in normrule.function.keys():
    #         #     coef = -coef
    #         #     b0 = -b0
    #         b0 = b0 - normrule.ub
    #         # print(len(zeroList),name2idx[yname])
    #         zeroList[name2idx[yname]] = coef
    #         # print(zeroList)
    #
    #         print(zeroList)
    #         res = linprog(c=-zeroList, A_ub=A, b_ub=b,options=options)
    #         print(res)
    #         Maxvalue = -res['fun'] + b0
    #         # print(Maxvalue)
    #         if Maxvalue <= 0 :
    #             continue
    #         else:
    #             print(Maxvalue)
    #
    #             print("Imp False")
    #             return  False
    # # print("Imp Yes")
    # return  True




def write2CSV(filename,head,D):
    with open('./ExperimentResult/'+filename , 'wt' ) as f2:
        cw = csv.writer(f2,lineterminator = '\n')
        cw.writerow(head)
        for item in D:
            cw.writerow(item)


if __name__ =="__main__":
    write2CSV("test.csv",["T1","T2"],[[1,2],[3,4]])
    # f0 = {}
    # f0["param"] = 'add(0, x)'
    # r0 = rule.rule(["x"], "y", f0, -20, 21, None)
    # f1 = {}
    # f1["param"] = 'add(0, x)'
    # r1 = rule.rule(["x"], "y", f1, -2, 1, None)
    # setR = [r1]
    # checkNonLinearImp(setR, r1, ["x", "y"], [0, 0])
    # rule 0 : y=x [-10,10]
    # f0 = {}
    # f0["Type"] = 'Linear'
    # f0["coef"] = np.array([1])
    # f0['intercept'] = 0
    # r0 = rule.rule(["0-0"],"0-1",f0,-10,10)
    # # rule 1 : y=x [-5,5]
    # f1 = {}
    # f1["Type"] = 'Linear'
    # f1["coef"] = np.array([1])
    # f1['intercept'] = 0
    # r1 = rule.rule(["0-0"], "0-1", f0, -25, 25)
    #
    # # it is obvioius that if r1 then r0
    # setR = [r1]
    # checkIfLinearImp(setR,r0)

    # p = DataLoader.DataLoader("NASDAQ")
    # p = p.file2Numpy()
    # x = xWindowize(p, 1)
    # x = xWindowize(p,2)
    # x = xWindowize(p, 3)

