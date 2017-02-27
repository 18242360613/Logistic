from numpy import *
import matplotlib.pyplot as plt

'''加载数据'''
def loadDataSet():
    fr = open('testSet.txt')
    fileLines = fr.readlines();
    dataMat = []; labelMat = [];
    for lines in fileLines:
        wordsArray = lines.strip().split()
        dataMat.append( [1.0,float(wordsArray[0]),float(wordsArray[1])] );
        labelMat.append(int(wordsArray[-1]))

    return dataMat,labelMat

'''定义跳跃函数'''
def sigmoid(x):
    try:
        1 + exp(-x)
    except Exception:
        print(x)

    return 1.0/(1 + exp(-x) )

'''梯度上升函数，利用函数梯度的方向是函数变化最快的方向进行求解
   比较容易实现，可能会陷入局部最优的情况
'''
def grandAsent(traMat,labelMat):

    dataMatrix = mat(traMat)#将数据转成矩阵
    labelMatrix = mat(labelMat).transpose()#标签矩阵转置

    m,n = shape( dataMatrix )#获得矩阵的行和列
    weights = ones((n,1))#初始权重设置为1
    alpha = 0.001;#设置步长
    maxCycles = 500;#迭代次数

    for i in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #计算当前权重时的结果
        error = labelMatrix - h#计算误差
        weights = weights + alpha*dataMatrix.transpose()*error#重新计算权重

    return  weights

'''随机梯度上升'''
def stocGradAscent(dataMatrix,labelMat):

    dataMatrix = array(dataMatrix) #从python自带的list转成numpy的array
    m,n = shape(dataMatrix)#获得数组的长度和宽度
    alpha = 0.01;
    weights = ones((n))
    for i in range(m):#随机梯度上升
        h = sigmoid(sum(dataMatrix[i]*weights));
        errors = labelMat[i] - h;
        weights = weights + alpha * errors * dataMatrix[i]

    return weights

def stocGradAscentImprove(dataMatrix,labelMat,numIter = 150):
    dataMatrix = array(dataMatrix)  # 从python自带的list转成numpy的array
    m, n = shape(dataMatrix)  # 获得数组的长度和宽度
    weights = ones((n))
    for i in range(numIter):  # 随机梯度上升
        dataIndex = list( range(m) )
        for j in range(m):
            alpha = 4/(i+j+1.0) + 0.0001;
            randomIndex = int( random.uniform(0,len(dataIndex)) )
            h = sigmoid(sum(dataMatrix[randomIndex] * weights ));
            errors = labelMat[randomIndex] - h;
            weights = weights + alpha * errors * dataMatrix[randomIndex]
            del(dataIndex[randomIndex])
    return weights

'''根据属性X1和属性X2的关系，画出分布图'''
def plotBestFit():
    dataMat,labelMat = loadDataSet();
#   weights = grandAsent(dataMat,labelMat)
#   weights = stocGradAscent(dataMat,labelMat)
    weights = stocGradAscentImprove(dataMat,labelMat)

    n = len(dataMat) #获得数据长度
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];

    for i in range(n):
        if labelMat[i] == 1:#根据数据属性类别，将数据归到不同的类别
            xcord1.append(dataMat[i][1]);
            ycord1.append(dataMat[i][2]);
        else:
            xcord2.append(dataMat[i][1]);
            ycord2.append(dataMat[i][2]);

    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')#类别1用红色的点标出
    ax.scatter(xcord2,ycord2,s=30,c='green')#类别2用绿色的点标出
    x = arange(-4.0,4.0,0.1)
    y = (-weights[0] - weights[1]*x)/weights[2] #假使y = 0,然后画出分类线
    y = mat(y).transpose() #将y装置，以符合分类条件

    ax.plot(x,y)
    plt.xlabel('X1');#添加横纵坐标名称
    plt.ylabel('X2');
    plt.show(); #显示图像

def classifyVector(intX,weights):

    prob = sigmoid(sum(intX*weights))
    if prob > 0.5: return 1.0;
    else: return  0.0;

def colicTest():
    frtrain = open('horseColicTraining.txt')
    traMat = [];labelMat = [];
    lines = frtrain.readlines();
    for line in lines:
        wordsarray = line.strip().split('\t');
        words = [];
        for i in range(21):
            words.append( float(wordsarray[i]) )
        traMat.append(words)
        labelMat.append( float(wordsarray[21]) )

    weights = stocGradAscentImprove(array(traMat),labelMat,numIter=500);

    frtest = open('horseColicTest.txt')
    errorCount = 0.0;numberTest = 0.0;
    testLines = frtest.readlines();
    for testline in testLines:
        numberTest += 1.0;
        currentline = testline.strip().split('\t');
        testdata = [];
        for i in range(21):
            testdata.append(float(currentline[i]))
        if( int(classifyVector(array(testdata),array(weights))) != int(currentline[-1]) ):
            errorCount += 1;

    errorRate = (float(errorCount) / numberTest)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 1; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print( "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)) )