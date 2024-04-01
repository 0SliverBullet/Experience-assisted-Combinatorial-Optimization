from math import exp
import time
import numpy as np
import os
POPSIZE=5    # the size of population
DIMENSION=0  # Number of facilities
NUMBER=0    # Number of customers
MAXITERA=0 # Number of iterations
RTIME=3     # Number of repeat runs
Ts=100
iter=5
cr=0.995
class indi:
    def __init__(self):
        self.x = []
        self.fitx=0.0
individual = []
BEST=indi()
Cost=[[]]
Fvalue=[]
# read instances from ORLIB dataset
def getData1(filepath):
    file=open(filepath)
    s=file.readline().split(" ")
    m,n=eval(s[1]),eval(s[2])
    f=np.zeros((m,))
    c=np.zeros((n,m))
    for j in range(m):
        s=file.readline().split(" ")
        f[j]=eval(s[2])
    for i in range(n):
        s=file.readline()
        count=0
        while count<m:
            s=file.readline().split(" ")
            for ss in s:
                if (ss=='') or (ss=='\n'):
                    continue
                c[i][count]=eval(ss)
                count+=1
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


def SA():
        t=Ts
        while (t>0.1):
                # print(t)
                t=t*cr
                curIter=0
                while curIter<iter:
                        curIter=curIter+1
                        p1 =np.random.randint(0, POPSIZE)
                        individual[POPSIZE].x=np.array(individual[p1].x)
                        rdft=np.random.randint(0, 7384)/ 7383.0
                        if rdft<0.4:                                 
                                 open_facility_indexes = np.where(individual[POPSIZE].x)[0]
                                 if (len(open_facility_indexes>0)):
                                        exchange2 =np.random.randint(0, len(open_facility_indexes))
                                        individual[POPSIZE].x[open_facility_indexes[exchange2]]=0 
                                        open_facility_indexes = np.where(1-individual[POPSIZE].x)[0]
                                        exchange1 =np.random.randint(0, len(open_facility_indexes))
                                        individual[POPSIZE].x[open_facility_indexes[exchange1]]=1
                                 else:
                                     curIter-=1
                                     continue
                        elif rdft<0.7:
                                 open_facility_indexes = np.where(1-individual[POPSIZE].x)[0]
                                 if (len(open_facility_indexes>0)):
                                        exchange1 =np.random.randint(0, len(open_facility_indexes))
                                        individual[POPSIZE].x[open_facility_indexes[exchange1]]=1
                                 else:
                                     curIter-=1
                                     continue
                        else:
                                 open_facility_indexes = np.where(individual[POPSIZE].x)[0]
                                 if (len(open_facility_indexes>0)):
                                        exchange2 =np.random.randint(0, len(open_facility_indexes))
                                        individual[POPSIZE].x[open_facility_indexes[exchange2]]=0   
                                 else:
                                     curIter-=1
                                     continue    
                        computeObjective(POPSIZE)  
                        if (individual[POPSIZE].fitx < individual[p1].fitx):
                            individual[p1].x =np.array( individual[POPSIZE].x)
                            individual[p1].fitx = individual[POPSIZE].fitx
                        else:
                             rdft=np.random.randint(0, 7384)/ 7383.0
                             if (rdft<exp(-(individual[POPSIZE].fitx - individual[p1].fitx)/t)):
                                individual[p1].x = np.array(individual[POPSIZE].x)
                                individual[p1].fitx = individual[POPSIZE].fitx 
                        if (individual[p1].fitx < BEST.fitx):
                                BEST.x=np.array(individual[p1].x)
                                BEST.fitx=individual[p1].fitx

def computeObjective(num):
    obj = 0.0
    obj += np.inner(individual[num].x, Fvalue)
    open_facility_indexes = np.where(individual[num].x)[0]
    if len(open_facility_indexes)==0:
         individual[num].fitx = 10000000000000000000.0
    else:
            temp = np.min(Cost[open_facility_indexes,:], axis=0)
            obj += np.sum(temp)
            individual[num].fitx = obj

def initialize():
    for i in range(POPSIZE):
        individual[i].x = np.random.randint(0, 2, DIMENSION)
        computeObjective(i)
    flg = 0
    for i in range(POPSIZE):
        if (individual[i].fitx < individual[flg].fitx):
            flg = i
    BEST.x=np.array(individual[flg].x)
    BEST.fitx=individual[flg].fitx
if __name__=='__main__':
    #benchmark
    cases=[] #(f,c,name,ans)
    paths=["./Expand/instances/ORLIB/ORLIB-uncap/70","./Expand/instances/ORLIB/ORLIB-uncap/100","./Expand/instances/ORLIB/ORLIB-uncap/130","./Expand/instances/ORLIB/ORLIB-uncap/a-c"]
    for filePath in paths:
        fcases=os.listdir(filePath)
        for pcase in fcases:
            if ('.opt' in pcase) or ('.lst' in pcase):
                continue
            f,c=getData1(filePath+"/"+pcase)
            v=open(filePath+"/"+pcase+".opt").readline().replace('\n','').split(" ")
            x=np.zeros(len(f))
            for j in range(len(v)-1):
                x[int(v[j])]=1
            cases.append((f,c,filePath+"/"+pcase,eval(v[len(v)-1]),x))
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
    generation_list=[
        1,1,1,1,
        1,1,1,1,
        1,1,1,1,
        4,4,4,
        2,2,2,2,2,
        3,3,3,3,3,
        4,4,4,4,4,
        5,5,5,5,5,
        8,
        10,

        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,

        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,

        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,

        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,  
        ]
    cnt=0
    for f,c,name,ans,x in cases:
        DIMENSION=len(f)
        NUMBER=len(c)
        Fvalue=f
        Cost=np.zeros((DIMENSION,NUMBER))
        for i in range(NUMBER):
                 for j in range(DIMENSION):
                     Cost[j][i]=c[i][j]
        data = np.arange(0,POPSIZE,1)
        data=data.tolist()
        print('filename',name)
        print('optimal value',ans)
        print('ans vector',x)
        name=name.split('/')[-1]
        name=name.split('.')[0]
        output_path = "./ESA/"+name+"_"+str(DIMENSION)+"X"+str(NUMBER)+"_ESA_F.txt"
        RT = 0
        MAXITERA=generation_list[cnt]
        individual.clear()
        with open(output_path, 'w', encoding='utf-8') as file2:
                    individual.clear()
                    for i in range(POPSIZE+1):
                            a=indi()
                            a.x=np.zeros((DIMENSION,))
                            individual.append(a)
                    BEST.x=np.zeros((DIMENSION,))
                    BEST.fitx=0.0
                    runtime=[]
                    best_list=[]
                    while RT < RTIME:
                        print(RT+1,file=file2)
                        FLAG = 0
                        initialize()
                        generation = 0
                        while (generation < MAXITERA ):
                            start1=time.time()
                            SA()
                            run_time1 = (time.time() - start1)
                            value=np.zeros((POPSIZE,))
                            for i in range(POPSIZE):
                                value[i]=individual[i].fitx
                            FLAG=FLAG+run_time1
                            print(generation,"%.5f" % BEST.fitx,file=file2)
                            generation=generation+1
                            print(generation)
                        print("%.5f" % BEST.fitx, "%.5f" % FLAG,file=file2)
                        best_list.append(BEST.fitx)
                        runtime.append(FLAG)
                        RT=RT+1
                        mean=np.mean(best_list)
                    gap=100*(mean-ans)/ans
                    print("%.5f" % mean,"%.5f" % np.std(best_list),"%.5f" % np.mean(runtime), "%.5f%%" % gap ,file=file2)
        file2.close()
        cnt+=1


            
