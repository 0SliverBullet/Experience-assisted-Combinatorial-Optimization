import os
import numpy as np
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
def loadInstances():
    cases=[]
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
    return cases
