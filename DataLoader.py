import pandas as pd
import numpy as np
import datetime
import time


class DataLoader:
    """
    主要职责：从数据集里面读取数据,返回numpy形式的数据结果
    """
    def __init__(self, name, trainOrTest="TRAIN"):
        """
        从数据集文件夹里面读取数据
        :param name: 文件夹的名字
        :param trainOrTest: 是训练集还是测试集
        """
        self.name = name
        self.trainOrTest = trainOrTest
        self.FileName = ""
        self.head =None

    def file2Numpy(self):
        """
        从文件夹里面读文件，并返回对应的numpy对象
        :return:
        """
        if self.name == "NASDAQ":
            f = open("./dataSet/STOCK/NASDAQ.csv")
            D = []
            row = []
            line = f.readline()
            line = line[:-1]
            idx = 1
            while line:
                line = f.readline()
                line = line[:-1]
                L = line.split(",")
                row = []
                flag = 0
                for strtmp in L:
                    if "/" in strtmp:
                        row.append(idx)
                        idx += 1
                        continue
                    else:
                        if strtmp == "":
                            flag = -1
                            break
                        row.append(float(strtmp))
                if flag == -1:
                    continue
                else:
                    D.append(row)
            D = np.array(D)
            # print(D.shape)
            return D


if __name__ =="__main__":
    DL = DataLoader("NASDAQ")
    A=DL.file2Numpy()
    print(A.shape)
    print(A)
