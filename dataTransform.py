import numpy as np
def Transform1():
    rown = 5000
    f = open("./dataSet/Wind/WindA.csv")
    D = []
    row = []
    line = f.readline()
    print(line)
    line = line[:-1]
    f1 = open("./dataSet/Wind/ContrastA.csv", "w")
    r=0
    oneHot = np.ones(39)
    idx=0
    while line:
        line = f.readline()
        idx+=1
        if idx % 10 == 1:
            str1 = line.split(",")
            str2 = ""
            for i0 in range(len(oneHot)):
                if i0 == 0:
                    continue
                elif i0 == len(str1)-1:
                    str2 += str1[i0]
                elif i0 == len(oneHot) - 1 and oneHot[i0] == 1:
                    str2 += str1[i0]
                    str2+="\n"
                else:
                    if oneHot[i0] == 1:
                        str2 += str1[i0]
                        str2+=","
            f1.write(str2)

    print(idx)
    D = np.array(D)
def Transform2():
    f = open("./dataSet/Wind/WindA.csv")
    D = []
    row = []
    line = f.readline()
    print(line)
    line = line[:-1]
    f1 = open("./dataSet/Wind/ContrastA.csv", "w")
    r = 0
    #,引风机3A总有功功率,引风机3A入口烟气压力,引风机3A出口烟气压力,引风机3A出口烟气温度,引风机3A动叶调节挡板控制指令,引风机3A电流,引风机3A入口烟气流量,引风机3A入口烟气流量,引风机3A动叶调节挡板阀位,引风机3A驱动端轴承温度1,引风机3A驱动端轴承温度2,引风机3A驱动端轴承温度3,引风机3A推力轴承温度1,引风机3A推力轴承温度2,引风机3A推力轴承温度3,引风机3A支撑轴承温度1,引风机3A支撑轴承温度2,引风机3A支撑轴承温度3,引风机3A非驱动端轴承温度1,引风机3A非驱动端轴承温度2,引风机3A非驱动端轴承温度3,引风机3A电机非驱动端轴承温度1,引风机3A电机非驱动端轴承温度2,引风机3A电机驱动端轴承温度1,引风机3A电机驱动端轴承温度2,引风机3A电机线圈绕组温度U1,引风机3A电机线圈绕组温度U2,引风机3A电机线圈绕组温度V1,引风机3A电机线圈绕组温度V2,引风机3A电机线圈绕组温度W1,引风机3A电机线圈绕组温度W2,引风机3A轴承振动X向1,引风机3A轴承振动X向2,引风机3A轴承振动Y向1,引风机3A轴承振动Y向2,引风机3A液压油压力,引风机3A润滑油压力,引风机3A油泵出口滤网差压,引风机3A润滑油进油管油温,引风机3A轴承润滑油回油管油温
    oneHot = np.zeros(39)
    NeedHead = ["引风机3A总有功功率","引风机3A非驱动端轴承温度2","引风机3A驱动端轴承温度1","引风机3A动叶调节挡板阀位","引风机3A电流","引风机3A出口烟气压力"]
    HeadStr = "引风机3A总有功功率,引风机3A入口烟气压力,引风机3A出口烟气压力,引风机3A出口烟气温度,引风机3A动叶调节挡板控制指令,引风机3A电流,引风机3A入口烟气流量,引风机3A入口烟气流量,引风机3A动叶调节挡板阀位,引风机3A驱动端轴承温度1,引风机3A驱动端轴承温度2,引风机3A驱动端轴承温度3,引风机3A推力轴承温度1,引风机3A推力轴承温度2,引风机3A推力轴承温度3,引风机3A支撑轴承温度1,引风机3A支撑轴承温度2,引风机3A支撑轴承温度3,引风机3A非驱动端轴承温度1,引风机3A非驱动端轴承温度2,引风机3A非驱动端轴承温度3,引风机3A电机非驱动端轴承温度1,引风机3A电机非驱动端轴承温度2,引风机3A电机驱动端轴承温度1,引风机3A电机驱动端轴承温度2,引风机3A电机线圈绕组温度U1,引风机3A电机线圈绕组温度U2,引风机3A电机线圈绕组温度V1,引风机3A电机线圈绕组温度V2,引风机3A电机线圈绕组温度W1,引风机3A电机线圈绕组温度W2,引风机3A轴承振动X向1,引风机3A轴承振动X向2,引风机3A轴承振动Y向1,引风机3A轴承振动Y向2,引风机3A液压油压力,引风机3A润滑油压力,引风机3A油泵出口滤网差压,引风机3A润滑油进油管油温,引风机3A轴承润滑油回油管油温"
    HeadList =HeadStr.split(",")
    Str2idx = {}
    idx=1

    # oneHot = np.ones(39)
    for h in HeadList:
        Str2idx[h] = idx
        idx+=1
    for selectStr in NeedHead:
        oneHot[ Str2idx[selectStr] ] = 1
    idx = 0
    oneHot[38] = 0
    while line:
        line = f.readline()
        line = line[:-1]
        idx += 1
        # if idx % 2 == 1:
        if True:
            str1 = line.split(",")
            str2 = ""
            for i0 in range(len(oneHot)):
                if i0 == 0:
                    continue
                elif i0 == len(str1) - 1:
                    str2 += str1[i0]
                elif i0 == len(oneHot) - 1 and oneHot[i0] == 1:
                    str2 += str1[i0]
                else:
                    if oneHot[i0] == 1:
                        str2 += str1[i0]
                        str2 += ","
            f1.write(str2+"\n")
    print(idx)
    D = np.array(D)

if __name__ == "__main__":
    Transform2()