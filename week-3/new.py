import os
import time
import random as ra
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pg
import sklearn.pipeline as pe
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset


# mRange is [mStart, mEnd]
# isE means is it euclidean space
def generateInstance(mRange,isE):
    # n/m from 1 to 10
    m=ra.randint(mRange[0],mRange[1])
    n=m*ra.randint(1,10)
    #c from 1 to 200, f/c from 1 to 20
    maxf=200*ra.randint(1,20)
    f=np.random.rand(m,)*(maxf-10)+10
    if(isE):
        c=np.zeros((n,m))
        #square with size(141，141)，the largest distance is 141*1.414=200
        mlocation=np.random.rand(m,2)*141
        nlocation=np.random.rand(n,2)*141
        for i in range(n):
            for j in range(m):
                c[i][j]=np.sqrt((nlocation[i][0]-mlocation[j][0])*(nlocation[i][0]-mlocation[j][0])+(nlocation[i][1]-mlocation[j][1])*(nlocation[i][1]-mlocation[j][1]))
        return f,c
    else:
        c=np.random.rand(n,m)*199+1
        return f,c
    
def solveInstances(f,c):
    #may be use optimal solver in the future
    print()

# read instances from OR dataset
def getData1(filepath):
    file=open(filepath)
    print(filepath)
    s=file.readline().split(" ")
    m,n=eval(s[1]),eval(s[1])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().split(" ")
        f[j]=eval(s[2])
    for i in range(n):
        s=file.readline()
        s=file.readline().split(" ")
        print(s)
        for j in range(m):
            c[i][j]=eval(s[j])
    return f,c

# read instances from M* dataset
def getData2(filepath):
    file=open(filepath)
    s=file.readline().split(" ")
    m,n=eval(s[0]),eval(s[1])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().split(" ")
        f[j]=eval(s[1])
    for i in range(n):
        s=file.readline()
        s=file.readline().split(" ")
        for j in range(m):
            c[i][j]=eval(s[j])
    return f,c

# read instances from Euclid,GapA,GapB,GapC dataset
def getdata3(filepath):
    file=open(filepath)
    s=file.readline()
    s=file.readline().split(" ")
    m,n=eval(s[0]),eval(s[1])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().replace('\n','').split(" ")
        f[j]=eval(s[1])
        for i in range(n):
            c[i][j]=s[i+2]
    return f,c

# not efficient in evolutionary algorithm, only used in test()
def functionValue(f,c,y):
    m=len(y)
    n=len(c)
    cSorted=[]
    for i in range(n):
        cSorted.append(np.argsort(c[i]))
    cSorted=np.array(cSorted)
    x=np.zeros((n,m))
    for i in range(n):
        for k in range(m):
            if(y[cSorted[i][k]]==1):
                x[i][cSorted[i][k]]=1
                break
    ans=np.sum(y*f)+np.sum(x*c)
    return ans

def test():
    # paths=["./Expand/instances/ORLIB/ORLIB-uncap/70","./Expand/instances/ORLIB/ORLIB-uncap/100","./Expand/instances/ORLIB/ORLIB-uncap/130","./Expand/instances/ORLIB/ORLIB-uncap/a-c"]
    # for filePath in paths:
    #     fcases=os.listdir(filePath)
    #     for pcase in fcases:
    #         if ('.opt' in pcase) or ('.lst' in pcase):
    #             continue
    #         else:
    #             f,c=getData1(filePath+"/"+pcase)
    #             v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
    #             x=np.zeros(len(f))
    #             for j in range(len(v)-1):\
    #                 x[int(v[j])]=1
    #             print(functionValue(f,c,x),v[len(v)-1])

    paths=["./Expand/instances/M/O","./Expand/instances/M/P","./Expand/instances/M/Q"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.opt' in pcase) or ('.lst' in pcase):
                continue
            else:
                f,c=getData2(filePath+"/"+pcase)
                v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
                x=np.zeros(len(f))
                for j in range(len(v)-1):
                    x[int(v[j])]=1
                print(filePath+"/"+pcase,functionValue(f,c,x),v[len(v)-1])

    paths=["./Expand/instances/M/R","./Expand/instances/M/S","./Expand/instances/M/T"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.bub' in pcase) or ('.lst' in pcase):
                continue
            else:
                f,c=getData2(filePath+"/"+pcase)
                v=open(filePath+"/"+pcase+".bub").readline().replace('\n','').split(" ")
                x=np.zeros(len(f))
                for j in range(len(v)-1):
                    x[int(v[j])]=1
                print(filePath+"/"+pcase,functionValue(f,c,x),v[len(v)-1])
    
    paths=["./Expand/instances/Euclid","./Expand/instances/GapA","./Expand/instances/GapB","./Expand/instances/GapC"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.opt') in pcase or ('.lst') in pcase:
                continue
            f,c=getdata3(filePath+"/"+pcase)
            v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
            x=np.zeros(len(f))
            for j in range(len(v)-1):
                x[int(v[j])]=1
            print(filePath+"/"+pcase,functionValue(f,c,x),v[len(v)-1])

#########################################################################################################################################

#!!!
#!!!EA needs improvement
#!!!

# the individuals in EA
class init():
    def __init__(self,y,f,c,cSorted):
        if np.sum(y)==0:
            y[int(ra.random()*len(y))]=1
        self.y=y
        self.value=self.getValue(y,self.getX(y,cSorted),f,c)
        self.s=str(y)
    def getX(self,y,cSorted):
        m=len(y)
        n=len(cSorted)
        x=np.zeros((n,m))
        for i in range(n):
            for k in range(m):
                if(y[cSorted[i][k]]==1):
                    x[i][cSorted[i][k]]=1
                    break
        return x
    def getValue(self,y,x,f,c):
        ans=np.sum(y*f)+np.sum(x*c)
        return ans

def getInitial(m,size,f,c,cSorted):
    pop=[]
    for j in range(size):
        y=np.random.randint(0,2,(m,))
        pop.append(init(y,f,c,cSorted))
    return pop

def mutation(pop,p,f,c,cSorted):
    off=[]
    m=len(pop[0].y)
    for i in pop:
        y=i.y.copy()
        for j in range(m):
            if(ra.random()<p):
                y[j]=1-y[j]
        off.append(init(y,f,c,cSorted))
    return off

def crossover(pop,size,f,c,cSorted):
    off=[]
    p=[]
    sum=0
    m=len(pop[0].y)
    for i in pop:
        sum+=1/i.value
        p.append(sum)
    l=len(p)
    for i in range(l):
        p[i]/=sum
    for i in range(size):
        a=None
        r=ra.random()
        for k in range(l):
            if r<p[k]:
                a=pop[k].y
                break
        b=None
        r=ra.random()
        for k in range(l):
            if r<p[k]:
                b=pop[k].y
                break
        if ((a is None) or (b is None)):
            continue
        mid=ra.randint(0,m-1)
        off.append(init(np.append(a[:mid],b[mid:]),f,c,cSorted))
        off.append(init(np.append(b[:mid],a[mid:]),f,c,cSorted))
    return off

def selection(pop,size):
    name={'123'}
    new=[]
    for i in pop:
        if(i.s not in name):
            name.add(i.s)
            new.append(i)
    pop=sorted(new,key=lambda init:init.value)
    if len(pop)>size:
        pop=pop[:size]
    return pop

def EA(f,c,size,generation):
    m=len(f)
    n=len(c)
    cSorted=[]
    for i in range(n):
        cSorted.append(np.argsort(c[i]))
    cSorted=np.array(cSorted)
    pop=getInitial(m,size,f,c,cSorted)
    generations=[]
    anss=[]
    for g in range(generation):
        off1=mutation(pop,0.1,f,c,cSorted)
        off2=crossover(pop,int(size/2),f,c,cSorted)
        pop=pop+off1+off2
        pop=selection(pop,size)
        generations.append(g)
        anss.append(pop[0].value)
    return pop,generations,anss

def getInitialH(y,size,f,c,cSorted):
    pop=[]
    pop.append(init(y,f,c,cSorted))
    m=len(y)
    for j in range(size-1):
        newY=y.copy()
        for j in range(m):
            if(ra.random()<0.1):
                newY[j]=1-newY[j]
        pop.append(init(newY,f,c,cSorted))
    return pop

def mutationH(pop,p,f,c,cSorted,h,EAHMatrix,model):
    off=[]
    m=len(pop[0].y)
    count0=0
    count1=1
    for i in pop:
        y=i.y.copy()
        for j in range(m):
            if(ra.random()<p):
                EAHVector=[]
                EAHVector.append(getSurVector(j,y,h,EAHMatrix))
                numbers,open = torch.max(model(torch.from_numpy(np.array(EAHVector).astype(np.float32))).data,1)
                if(open==1):
                    count1+=1
                else:
                    count0+=1
                y[j]=open.item()
        off.append(init(y,f,c,cSorted))
    print(count0,count1)
    return off

def EAH(y,f,c,size,generation,muH,model):
    m=len(f)
    n=len(c)
    cSorted=[]
    for i in range(n):
        cSorted.append(np.argsort(c[i]))
    cSorted=np.array(cSorted)
    h=getLAC(f,c)
    EAHMatrix=getSurMatrix(f,c)
    if y is None:
        pop=getInitial(m,size,f,c,cSorted)
    else:
        pop=getInitialH(y,size,f,c,cSorted)
    generations=[]
    anss=[]
    time1=0
    time2=0
    time3=0
    time4=0
    for g in range(generation):
        t1=time.time()
        off1=mutation(pop,0.1,f,c,cSorted)
        t2=time.time()
        off2=crossover(pop,int(size/2),f,c,cSorted)
        t3=time.time()
        if(muH):
            off3=mutationH(pop,0.1,f,c,cSorted,h,EAHMatrix,model)
            t4=time.time()
            pop=pop+off1+off2+off3
        else:
            pop=pop+off1+off2
        pop=selection(pop,size)
        t5=time.time()
        generations.append(g)
        anss.append(pop[0].value)
        time1+=t2-t1
        time2+=t3-t2
        time3+=t4-t3
        time4+=t5-t4
        #print(pop[0].value)
    print(time1,time2,time3,time4)
    return pop,generations,anss
######################################################################################################################

# the model
def ploynommial(x,y):
    model = pe.make_pipeline(
        pg.PolynomialFeatures(10), 
        lm.LinearRegression()
        )
    model.fit(x, y)
    return model

# # get opening condition Y from opening probability P
def getY(p):
    y=[]
    for i in range(len(p)):
        if(p[i]<0.5):
            y.append(0)
        else:
            y.append(1)
    return np.array(y)

# def getY(p,f,c,cSorted):
#     pop=[]
#     for q in range(99):
#         qq=(q+1)/100
#         y=[]
#         for i in range(len(p)):
#             if(p[i]<qq):
#                 y.append(0)
#             else:
#                 y.append(1)
#         pop.append(init(np.array(y),f,c,cSorted))
#     return selection(pop,1)[0].y

# get facility opening probability
def getP(m,pop):
    p=np.zeros((m,))
    for i in pop:
        p+=i.y
    p/=len(pop)
    return p

# get local apportioned cost
# def getLAC(f,c):
#     m=len(f)
#     n=len(c)
#     h=f.copy()
#     count=np.ones((m,))
#     for i in range(n):
#         min=float('inf')
#         index=-1
#         for j in range(m):
#             if(c[i][j]<min):
#                 index=j
#                 min=c[i][j]
#         h[index]+=min
#         count[index]+=1
#     hmax=-float('inf')
#     hmin=float('inf')
#     for j in range(m):
#         if(count[j]==0):
#             continue
#         h[j]=h[j]/count[j]
#         if h[j]>hmax:
#             hmax=h[j]
#         if h[j]<hmin:
#             hmin=h[j]
#     for j in range(m):
#         if(count[j]==0):
#             h[j]=2
#             continue
#         if hmin==hmax:
#             h[j]=2
#         else:
#             h[j]=(h[j]-hmin)/(hmax-hmin)
#     return h

def getLAC(f,c):
    m=len(f)
    n=len(c)
    h=f.copy()
    for i in range(n):
        for j in range(m):
            h[j]+=c[i][j]/((j+1)*(j+1))
    hmax=-float('inf')
    hmin=float('inf')
    for j in range(m):
        if h[j]>hmax:
            hmax=h[j]
        if h[j]<hmin:
            hmin=h[j]
    for j in range(m):
        if hmin==hmax:
            h[j]=2
        else:
            h[j]=(h[j]-hmin)/(hmax-hmin)
    return h

def trainLAC(cases,isE,name):
    h=[]
    p=[]
    for t in range(cases):
        f,c=generateInstance([5,30],isE)
        pop,generations,changes=EA(f,c,len(f),30)
        p=np.append(p,getP(len(f),pop))
        h=np.append(h,getLAC(f,c))
    model=ploynommial(np.array(h).reshape(-1,1),np.array(p).reshape(-1,1))
    pickle.dump(model,open(name,'wb'))
    return model

# the facility opening estimation heuristic, build solution from f,c,model
def FOE(f,c,model):
    h=getLAC(f,c)
    p=model.predict(np.array(h).reshape(-1,1))
    cSorted=[]
    for i in range(len(c)):
        cSorted.append(np.argsort(c[i]))
    cSorted=np.array(cSorted)
    
    #!!!
    #change the get Y, (p=0.1-0.99;p<0.3 0, 0.3<p<0.7 random, p>0.7 1; )
    #!!!
    return getY(p)

def getLACVector(f,c):
    hVector=[]
    m=len(f)
    n=len(c)
    Hnum=5
    Knum=3
    h=getLAC(f,c)
    for j in range(m):
        a=[]
        for i in range(n):
            a.append(c[i][j])
        a=np.argsort(a)
        d=np.zeros((m,))
        for l in range(m):
            for k in range(Knum):
                d[l]+=c[a[k]][l]
            d[l]/=Knum
        d[j]=float('inf')
        d=np.argsort(d)
        ve=[]
        ve.append(h[j])
        for i in range(Hnum-1):
            ve.append(h[d[i]])
        hVector.append(ve)
    return hVector

def trainLACVector(cases,isE):
    hVector=[]
    p=[]
    for t in range(cases):
        f,c=generateInstance(5,30,isE)
        pop,generations,changes=EA(f,c,len(f),30)
        p=np.append(p,getP(len(f),pop))
        hVector=hVector+getLACVector(f,c)
    model=ploynommial(np.array(hVector),np.array(p).reshape(-1,1))
    pickle.dump(model,open('modelHVector.sav','wb'))
    return model

def predictLACVector(f,c,cSorted,model):
    hVector=getLACVector(f,c)
    p=model.predict(np.array(hVector))
    a=init(getY(p),f,c,cSorted)
    return a.value

def getSurMatrix(f,c):
    Hnum=5
    Knum=3
    SurMatrix=np.zeros((len(f),Hnum-1))
    m=len(f)
    n=len(c)
    for j in range(m):
        a=[]
        for i in range(n):
            a.append(c[i][j])
        a=np.argsort(a)
        d=np.zeros((m,))
        for l in range(m):
            for k in range(Knum):
                d[l]+=c[a[k]][l]
            d[l]/=Knum
        d[j]=float('inf')
        d=np.argsort(d)
        for i in range(Hnum-1):
            SurMatrix[j][i]=d[i]
    return SurMatrix.astype(int)

def getSurVector(j,y,h,SurMatrix):
    ve=[]
    ve.append(h[j])
    for i in range(len(SurMatrix[j])):
        ve.append(h[SurMatrix[j][i]])
        ve.append(y[SurMatrix[j][i]])
    return ve

# def trainSurModel(feature,label):
#     feature=feature.astype(np.float32)
#     label=np.array(getY(label)).astype(np.longlong)
#     train_features = feature[:int(len(feature)*0.8)]
#     test_features = feature[int(len(feature)*0.8):]
#     train_label = label[:int(len(feature)*0.8)]
#     test_label = label[int(len(feature)*0.8):]
#     batch_size=32
#     train_dataset = dataset(train_features,train_label)
#     train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
#     test_dataset = dataset(test_features,test_label)
#     test_loader = torch.utils.data.DataLoader(dataset = test_dataset,shuffle = True)
#     model=Model()
#     criterion =torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
#     for epoch in range(1000):
#         model.train()
#         ans = 0
#         for step,data in enumerate(train_loader):
#             x,y=data
#             loss = criterion(model.forward(x), y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             ans += loss.item()
#             if step % 100 == 0:
#                 print('[%d, %5d] loss: %.3f' %(epoch + 1, step, ans / 100))
#                 ans = 0
#     ans = 0
#     legnth = 0
#     with torch.no_grad():
#         model.eval()
#         for (images,labels) in test_loader:
#             numbers,predicted = torch.max(model(images).data,1)
#             legnth +=labels.size(0)
#             ans+=(predicted==labels).sum().item()
#     print('Accuracy: ',ans / legnth)
#     torch.save(model.state_dict(),'para.pth')
#     return model

class dataset(Dataset):
    def __init__(self, features, target):
        self.features = torch.from_numpy(features)
        self.target = torch.from_numpy(target)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.target[index]

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(9,108)
        self.fc2 = torch.nn.Linear(108,72)
        self.fc3 = torch.nn.Linear(72,18)
        self.fc4 = torch.nn.Linear(18,2)
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x_1 = torch.sigmoid(self.fc1(x))
        x_2 = torch.sigmoid(self.fc2(x_1))
        x_3 = torch.sigmoid(self.fc3(x_2))
        x_out = F.softmax(self.fc4(x_3),1)
        return x_out

def trainSurModel(feature,label):
    feature=feature.astype(np.float32)
    label=label.astype(np.float32)
    train_features = feature[:int(len(feature)*0.8)]
    test_features = feature[int(len(feature)*0.8):]
    train_label = label[:int(len(feature)*0.8)]
    test_label = label[int(len(feature)*0.8):]
    batch_size=1
    train_dataset = dataset(train_features,train_label)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
    test_dataset = dataset(test_features,test_label)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=1,shuffle = True)
    model=Model()
    criterion =torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    for epoch in range(100):
        model.train()
        ans = 0
        for step,data in enumerate(train_loader):
            x,y=data
            optimizer.zero_grad()
            loss = criterion(model.forward(x), y)
            loss.backward()
            optimizer.step()
            ans += loss.item()
            # if step % 100 == 0:
            #     print('[%d, %5d] loss: %.3f' %(epoch + 1, step, ans / 100))
            #     ans = 0
        print(model.fc1.weight)
        print(model.fc1.bias)

    ans = 0
    legnth = 0
    pcount0=0
    pcount1=0
    tcount0=0
    tcount1=0
    with torch.no_grad():
        model.eval()
        for (images,labels) in test_loader:
            predicted = torch.argmax(model(images))
            legnth +=1
            if labels[0][predicted]==1:
                ans+=1
            if predicted==0:
                pcount0+=1
            else:
                pcount1+=1
            if labels[0][0]==1:
                tcount0+=1
            else:
                tcount1+=1
    print('Accuracy: ',ans / legnth)
    print(pcount0,pcount1,tcount0,tcount1)
    torch.save(model.state_dict(),'para.pth')
    return model
    
def trainSurVector(cases,isE):
    EAHVector=[]
    pEAH=[]
    count0=0
    count1=0
    for t in range(cases):
        f,c=generateInstance(5,30,isE)
        pop,generations,changes=EA(f,c,len(f),30)
        p=getP(len(f),pop)
        h=getLAC(f,c)
        EAHMatrix=getSurMatrix(f,c)
        for j in range(len(f)):
            if(p[j]>=0.5)or(count1>count0):
                if(p[j]>=0.5):
                    count1+=1
                else:
                    count0+=1
                for i in range(int(len(pop)/5)):
                    EAHVector.append(getSurVector(j,pop[i].y,h,EAHMatrix))
                    pEAH.append(p[j])
    model=trainSurModel(np.array(EAHVector),np.array(pEAH))
    pickle.dump(model,open('modelEAH.sav','wb'))
    return model

def train(cases,isE):
    h=[]
    hVector=[]
    p=[]
    EAHVector=[]
    EAHLabel=[]
    for t in range(cases):
        f,c=generateInstance([5,30],isE)
        pop,generations,changes=EA(f,c,len(f),30)
        ptem=getP(len(f),pop)
        p=np.append(p,ptem)
        htem=getLAC(f,c)
        h=np.append(h,htem)
        hVector=hVector+getLACVector(f,c)
        # EAHMatrix=getSurMatrix(f,c)
        # for j in range(len(f)):
        #     for i in range(int(len(pop)/5)):
        #         EAHVector.append(getSurVector(j,pop[i].y,htem,EAHMatrix))
        #         l=[0,0]
        #         l[pop[i].y[j]]=1
        #         EAHLabel.append(l)
        print(t)
    modelH=ploynommial(np.array(h).reshape(-1,1),np.array(p).reshape(-1,1))
    pickle.dump(modelH,open('./Expand/modelLAC.sav','wb'))
    modelHVector=ploynommial(np.array(hVector),np.array(p).reshape(-1,1))
    pickle.dump(modelHVector,open('./Expand/modelLACVector.sav','wb'))
    #modelEAH=trainSurModel(np.array(EAHVector),np.array(EAHLabel))
    #pickle.dump(modelEAH,open('./Expand/modelSurVector.sav','wb'))
    return modelH,modelHVector

def OPESurVector():

    #!!!
    #use in many steps
    #SA temperature, p>0.7 increases to p>0.9
    #!!!
    print()

def FOESurVector(f,c,modelLAC,modelLACVector,modelSurVector):
    y=np.zeros(len(f))
    h=getLAC(f,c)
    p=modelLAC.predict(np.array(h).reshape(-1,1))
    for i in range(len(f)):
        if p[i]>0.8:
            y[i]=1
    
    count=0
    print('begin')
    EAHMatrix=getSurMatrix(f,c)
    while count<10:

        total=0
        change=0
        for j in range(len(f)):
            EAHVector=[]
            EAHVector.append(getSurVector(j,y,h,EAHMatrix))
            predict = modelSurVector(torch.from_numpy(np.array(EAHVector).astype(np.float32))).data,1
            numbers,open = torch.max(modelSurVector(torch.from_numpy(np.array(EAHVector).astype(np.float32))).data,1)
            print(predict[0][0][0],open)
        if predict >0.9:
            y[j]=1
            change+=1
        total+=1
        if change/total<0.05:
            count+=1
        else:
            count=0
    print('end')



if __name__=='__main__':
    # j is facilities, total number is m
    # i is customers, total number is n
    # f[j] is opening cost
    # c[i][j] is service cost
    # 为了防止出错, 建议遍历 facilities 时都用 j , 遍历 customers 时都用 i.
    # 用 numpy 代替 for循环 提高效率，想到什么随时在群里面说，这种行为保持统一
    # dataPath='./instances/M/MO1/MO1'
    # f,c=getData1(dataPath)
    # m=len(f)
    # n=len(c)

    # train the model from 'cases' small cases
    # isE means is it euclidean space
    #train(10000,True)

    modelLAC=pickle.load(open('./Expand/modelLAC.sav','rb'))
    modelLACVector=pickle.load(open('./Expand/modelLACVector.sav','rb'))
    modelSurVector=pickle.load(open('./Expand/modelSurVector.sav','rb'))
    print('load over')
    
    #benchmark
    cases=[] #(f,c,name,ans)
    paths=["./Expand/instances/M/O","./Expand/instances/M/P","./Expand/instances/M/Q"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.opt' in pcase) or ('.lst' in pcase):
                continue
            f,c=getData2(filePath+"/"+pcase)
            v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
            x=np.zeros(len(f))
            for j in range(len(v)-1):
                x[int(v[j])]=1
            cases.append((f,c,filePath+"/"+pcase,eval(v[len(v)-1]),x))
    paths=["./Expand/instances/M/R","./Expand/instances/M/S","./Expand/instances/M/T"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.bub' in pcase) or ('.lst' in pcase):
                continue
            f,c=getData2(filePath+"/"+pcase)
            v=open(filePath+"/"+pcase+".bub").readline().replace('\n','').split(" ")
            x=np.zeros(len(f))
            for j in range(len(v)-1):
                x[int(v[j])]=1
            cases.append((f,c,filePath+"/"+pcase,eval(v[len(v)-1]),x))
    paths=["./Expand/instances/Euclid","./Expand/instances/GapA","./Expand/instances/GapB","./Expand/instances/GapC"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.opt') in pcase or ('.lst') in pcase:
                continue
            f,c=getdata3(filePath+"/"+pcase)
            v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
            x=np.zeros(len(f))
            for j in range(len(v)-1):
                x[int(v[j])]=1
            cases.append((f,c,filePath+"/"+pcase,eval(v[len(v)-1]),x))
    print('read over.')

    
    # wr=open('./Expand/LACinformation2.txt','w')
    # #打印LAC
    # for f,c,name,ans,x in cases:
    #     #get LAC
    #     m=len(f)
    #     n=len(c)
    #     h=f.copy()
    #     count=np.zeros((m,))
    #     hassign=[[] for j in range(m)]
    #     for i in range(n):
    #         min=float('inf')
    #         index=-1
    #         for j in range(m):
    #             if(c[i][j]<min):
    #                 index=j
    #                 min=c[i][j]
    #         h[index]+=min
    #         count[index]+=1
    #         hassign[index].append(i)
    #     hmax=-float('inf')
    #     hmin=float('inf')
    #     for j in range(m):
    #         if(count[j]==0):
    #             continue
    #         h[j]=h[j]/count[j]
    #         if h[j]>hmax:
    #             hmax=h[j]
    #         if h[j]<hmin:
    #             hmin=h[j]
    #     htem=h.copy()
    #     for j in range(m):
    #         if(count[j]==0):
    #             h[j]=2
    #             continue
    #         if hmin==hmax:
    #             h[j]=2
    #         else:
    #             h[j]=(h[j]-hmin)/(hmax-hmin)
    #     p=modelLAC.predict(np.array(h).reshape(-1,1))
    #     y=getY(p)
    #     information=[]
    #     for j in range(m):
    #         information.append((j,count[j],f[j],'\th,p:',h[j],p[j][0],'y:',y[j],x[j],'LAC,c:',htem[j]))
    #     information=sorted(information,key=lambda ii:-ii[5])
    #     wr.write('\n'+name+'\n')
    #     cSorted=[]
    #     for i in range(len(c)):
    #         cSorted.append(np.argsort(c[i]))
    #     for j in range(m):
    #         for k in information[j]:
    #             wr.write(str(k)+' ')
    #         for i in hassign[information[j][0]]:
    #             wr.write(str(c[i][information[j][0]])+' ')
    #         wr.write('\n')
    #     wr.flush()
    # wr.close()

    cc=0
    count=[]
    ansRandom=[]
    ansLAC=[]
    ansLACVector=[]
    ansSurVector=[]
    ansAns=[]
    for f,c,name,ans,x in cases:
        cSorted=[]
        for i in range(len(c)):
            cSorted.append(np.argsort(c[i]))
        cc+=1
        count.append(cc)
        cSorted=np.array(cSorted)
        r=selection(getInitial(len(f),100,f,c,cSorted),1)[0].value
        #sur=init(FOESurVector(f,c,modelLAC,modelLACVector,modelSurVector)).value
        ansRandom.append(r)
        ansLAC.append(init(FOE(f,c,modelLAC),f,c,cSorted).value)
        ansLACVector.append(predictLACVector(f,c,cSorted,modelLACVector))
        #ansSurVector.append(sur)
        ansAns.append(ans)
    plt.plot(count,ansRandom,'c',label='random')
    plt.plot(count,ansLAC,'g',label='LAC')
    plt.plot(count,ansLACVector,'b',label='LACVector')
    #plt.plot(count,ansSurVector,'y',label='SurVector')
    plt.plot(count,ansAns,'r',label='best')
    plt.title('the effect of solutions of different methods')
    plt.xlabel('instances')
    plt.ylabel('objective value')
    plt.legend()
    plt.savefig('./Expand/tem.png')
    plt.show()
    


            
