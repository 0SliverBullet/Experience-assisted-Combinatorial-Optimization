import os
import random as ra
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random as ra
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
import loadInstances
import numpy as np
import numpy.random as ra
feature=[]
label=[]
def solveInstances(f,c,generation):
    from solveInstancesDPSO import Main
    main_instance = Main(f,c,generation)
    anss = main_instance.DPSO()
    return anss
# mRange is [mStart, mEnd]
# isE means is it euclidean space

def generateInstance(mRange,isE):
    # n/m from 1 to 10
    m=ra.randint(mRange[0],mRange[1])
    n=m*ra.randint(1,10)
    #c from 1 to 200, f/c from 1 to 20
    p=ra.random()
    #print(p)
    fc_Vector=[]
    if p>0.5:
        maxf=200*ra.randint(1,20)
    else:
        maxf=200/ra.randint(1,20)
    f=np.random.rand(m,)*(maxf)
    if(isE):
        c=np.zeros((n,m))
        #square with size(141，141)，the largest distance is 141*1.414=200
        mlocation=np.random.rand(m,2)*141
        nlocation=np.random.rand(n,2)*141
        max_c=[]
        min_c=[]
        general_c1=[]
        general_c2=[]
        general_c3=[]
        general_c4=[]
        general_c5=[]
        general_c6=[]
        general_c7=[]
        general_c8=[]
        general_c9=[]
        for i in range(n):
            cost=[]
            for j in range(m):
                c[i][j]=np.sqrt((nlocation[i][0]-mlocation[j][0])*(nlocation[i][0]-mlocation[j][0])+(nlocation[i][1]-mlocation[j][1])*(nlocation[i][1]-mlocation[j][1]))
                cost.append(c[i][j])
            max_c.append(max(cost))
            min_c.append(min(cost))
            cost=np.sort(cost)
            k=len(cost)//10
            general_c1.append(cost[k])
            general_c2.append(cost[2*k])
            general_c3.append(cost[3*k])
            general_c4.append(cost[4*k])
            general_c5.append(cost[5*k])
            general_c6.append(cost[6*k])
            general_c7.append(cost[7*k])
            general_c8.append(cost[8*k])
            general_c9.append(cost[9*k])
        fc_Vector.append(np.mean(max_c)/np.mean(f))
        fc_Vector.append(np.mean(min_c)/np.mean(f))
        fc_Vector.append(np.mean(general_c1)/np.mean(f))
        fc_Vector.append(np.mean(general_c2)/np.mean(f))
        fc_Vector.append(np.mean(general_c3)/np.mean(f))
        fc_Vector.append(np.mean(general_c4)/np.mean(f))
        fc_Vector.append(np.mean(general_c5)/np.mean(f))
        fc_Vector.append(np.mean(general_c6)/np.mean(f))
        fc_Vector.append(np.mean(general_c7)/np.mean(f))
        fc_Vector.append(np.mean(general_c8)/np.mean(f))
        fc_Vector.append(np.mean(general_c9)/np.mean(f))
        return fc_Vector,f,c
    else:
        c=np.random.rand(n,m)*199+1
        return f,c

def train(cases,isE):
    feature=[]
    label=[]
    for t in range(cases):
        print(t)
        fc_Vector,f,c=generateInstance([5,50],isE)
        feature.append(fc_Vector)
        BEST=solveInstances(f,c,30)
        #label.append([sum(BEST)/len(BEST)])
        label.append([sum(BEST)])

    feature=np.array(feature).astype(np.float32)
    label=np.array(label).astype(np.float32)
    train_features = feature[:int(len(feature)*0.9)]
    test_features = feature[int(len(feature)*0.9):]
    train_label = label[:int(len(feature)*0.9)]
    test_label = label[int(len(feature)*0.9):]

    model = svm.SVR(kernel ='rbf', degree = 2, gamma ='auto', coef0 = 0.0, tol = 0.1, C = 1.0,
                     epsilon = 0.1, shrinking = True, cache_size = 200, verbose = False, max_iter = -1 )

    # 训练模型
    model.fit(train_features, train_label)
    # 预测测试数据
    y_pred = model.predict(test_features)

    y_test=test_label

    # 模型评估
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("均方误差：", mse)
    print("R2 分数：", r2)

    y_pred = model.predict(train_features)
    y_test=train_label
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("均方误差：", mse)
    print("R2 分数：", r2)    
    pickle.dump(model,open('./fc_open_number01.sav','wb'))


    return




if __name__=='__main__':
    # j is facilities, total number is m
    # i is customers, total number is n
    # f[j] is opening cost
    # c[i][j] is service cost
    # 遍历 facilities 时都用 j , 遍历 customers 时都用 i.
    # m=len(f)
    # n=len(c)

    # train the model from 'cases' small cases
    # isE means is it euclidean space
    
    train(5000,True)
    
    modelFCON=pickle.load(open('./fc_open_number01.sav','rb'))
    print('load over')
    
    #benchmark
    cases=loadInstances.loadInstances()
    
    cc=0
    count=[]
    ansFCON=[]
    ansAns=[]
    for f,c,name,ans,x in cases:
        cSorted=[]
        for i in range(len(c)):
            cSorted.append(np.argsort(c[i]))
        cc+=1
        count.append(cc)
        cSorted=np.array(cSorted)
        max_c=[]
        min_c=[]
        general_c1=[]
        general_c2=[]
        general_c3=[]
        general_c4=[]
        general_c5=[]
        general_c6=[]
        general_c7=[]
        general_c8=[]
        general_c9=[]
        n=len(c)
        m=len(f)
        fc_Vector=[]
        feature=[]
        for i in range(n):
            cost=[]
            for j in range(m):
                cost.append(c[i][j])
            max_c.append(max(cost))
            min_c.append(min(cost))
            cost=np.sort(cost)
            k=len(cost)//10
            general_c1.append(cost[k])
            general_c2.append(cost[2*k])
            general_c3.append(cost[3*k])
            general_c4.append(cost[4*k])
            general_c5.append(cost[5*k])
            general_c6.append(cost[6*k])
            general_c7.append(cost[7*k])
            general_c8.append(cost[8*k])
            general_c9.append(cost[9*k])
            # general_c.append(find_best_turning_point2(cost))
        fc_Vector.append(np.mean(max_c)/np.mean(f))
        fc_Vector.append(np.mean(min_c)/np.mean(f))
        # fc_Vector.append(np.mean(general_c)/np.mean(f))
        fc_Vector.append(np.mean(general_c1)/np.mean(f))
        fc_Vector.append(np.mean(general_c2)/np.mean(f))
        fc_Vector.append(np.mean(general_c3)/np.mean(f))
        fc_Vector.append(np.mean(general_c4)/np.mean(f))
        fc_Vector.append(np.mean(general_c5)/np.mean(f))
        fc_Vector.append(np.mean(general_c6)/np.mean(f))
        fc_Vector.append(np.mean(general_c7)/np.mean(f))
        fc_Vector.append(np.mean(general_c8)/np.mean(f))
        fc_Vector.append(np.mean(general_c9)/np.mean(f))
        feature.append(fc_Vector)
        feature=np.array(feature).astype(np.float32)
        print(feature)
        solution=modelFCON.predict(feature)
        ansFCON.append(solution)
        #ansAns.append(sum(x)/len(x))
        ansAns.append(sum(x))
    plt.plot(count,ansFCON,'g',label='FCON')
    plt.plot(count,ansAns,'r',label='best')
    plt.title('the effect of FCON in OR, M, Euclid, GapA, GapB, GapC')
    plt.xlabel('instances')
    plt.ylabel('open number')
    plt.legend()
    #plt.savefig('./Expand/Modified.png')
    plt.show()