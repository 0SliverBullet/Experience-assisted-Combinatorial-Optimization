import time
import numpy as np
import random
import os

POPSIZE=0   # the size of population
DIMENSION=0  # Number of facilities
NUMBER=0    # Number of customers
MAXITERA=0 # Number of iterations
MX=2          #[n]=[MX]={0,1,..,MX-1}={0,1}
Pm=0.02       #Mutation probability
RTIME=3      # Number of repeat runs
class indi:
    def __init__(self):
        self.x = []
        self.fitx=0.0

individual = []
BEST=indi()
Cost=[[]]
Fvalue=[]
data=[] #= np.arange(0,POPSIZE,1)
#global data=data.tolist()
# read instances from ORLIB dataset
# read instances from OR dataset
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
# def getData1(filepath):
#     file=open(filepath)
#     global DIMENSION
#     global NUMBER
#     global data
#     global POPSIZE
#     s=file.readline().split(" ")
#     DIMENSION,NUMBER=eval(s[0]),eval(s[1])
#     POPSIZE=DIMENSION 
#     data = np.arange(0,POPSIZE,1)
#     data=data.tolist()
#     global Fvalue
#     Fvalue=np.zeros((DIMENSION,))
#     global Cost
#     Cost=np.zeros((DIMENSION,NUMBER))
#     for j in range(DIMENSION):
#         s=file.readline().split(" ")
#         Fvalue[j]=eval(s[1])
#     for j in  range(NUMBER):
#         s=file.readline().split(" ")
#         s=file.readline().split(" ")
#         for i in range(DIMENSION):
#             Cost[i][j]=eval(s[i])
def updatefunction(xp1,  xp2,  xp3):
    temp=0
    if (xp2 == xp3):
        temp = 0
    else:
        temp = 1
    rdft=np.random.randint(0, 7384)/ 7383.0
    if (rdft < 0.5):
        return ((xp1 + temp) % MX) #FS=1
    else:
        return (xp1); #FS=0
def mutatefunction(num):
    for j in range(DIMENSION):
        rdft=np.random.randint(0, 7384)/ 7383.0
        if (rdft< Pm):
            individual[num].x[j] = 1 - individual[num].x[j]
def evolution():
    for i in range(POPSIZE):
        t = random.sample(data, k=3)
        for j in range(DIMENSION):
            individual[POPSIZE].x[j] = updatefunction(individual[t[0]].x[j], individual[t[1]].x[j], individual[t[2]].x[j])
        mutatefunction(POPSIZE)
        open_facility_indexes = np.where(individual[POPSIZE].x)[0]
        for j in open_facility_indexes:
            rdft=np.random.randint(0, 7384)/ 7383.0
            if rdft < 0.2:
                individual[POPSIZE].x[j] = 0 
        computeObjective(POPSIZE)  
        if (individual[POPSIZE].fitx < individual[i].fitx):
            individual[i].x=np.array(individual[POPSIZE].x)
            individual[i].fitx = individual[POPSIZE].fitx
            if (individual[i].fitx < BEST.fitx):
                 BEST.x=np.array(individual[i].x)
                 BEST.fitx=individual[i].fitx
def computeObjective(num):
    obj = 0.0
    open_facility_indexes = np.where(individual[num].x)[0]
    if len(open_facility_indexes)==0:
         individual[num].fitx = 10000000000000000000.0
    else:
            temp = np.min(Cost[open_facility_indexes,:], axis=0)
            for _ in np.unique(np.argmin(Cost[open_facility_indexes,:],axis=0)):
                obj+=Fvalue[open_facility_indexes[_]]
            obj +=np.sum(temp)
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
        150,150,150,150,
        150,150,150,150,
        150,150,150,150,
        200,200,200,
        150,150,150,150,150,
        300,300,300,300,300,
        200,200,200,200,200,
        250,250,250,250,250,
        260,
        300,

        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,

        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,

        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,

        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        150,150,150,150,150,
        
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
        POPSIZE=DIMENSION 
        data = np.arange(0,POPSIZE,1)
        data=data.tolist()
        print('filename',name)
        print('optimal value',ans)
        print('ans vector',x)
        name=name.split('/')[-1]
        name=name.split('.')[0]
        output_path = "./EGTOA/"+name+"_"+str(DIMENSION)+"X"+str(NUMBER)+"_EGTOA_N.txt"
        
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
                            evolution()
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



            
