import numpy as np


class rule:
    def __init__(self,xnameL,yname,func,lb,ub,eval):
        self.Xnamelist = xnameL
        self.eval = eval
        self.Yname = yname
        self.function = func
        self.lb = lb
        self.ub = ub
    def checkIfVaild(self,newX,newY):
        """
        Untested!
        :param newX:
        :param newY:
        :return:
        """
        return np.logical_and( (self.eval.predict(newX)-newY)> self.ub,(self.eval.predict(newX)-newY) < self.lb  )


    def normalize(self):
        f0 = self.function.copy()
        f0["neg"] = True
        rule1 = rule(self.Xnamelist,self.Yname,f0,lb="-inf",ub=-self.lb,eval=self.eval)
        rule2 = rule(self.Xnamelist, self.Yname, self.function, lb="-inf", ub=self.ub,eval=self.eval)
        return [rule1,rule2]

    def demo(self):
        print("-----Rule-----")
        print("Y:",self.Yname," <- " ,self.Xnamelist)
        print("Param:",self.function)
        print("lb:",self.lb,"\tub:",self.ub)

    def setbound(self,lb,ub):
        self.lb = lb
        self.ub = ub

