from __future__ import print_function
import cv2
import pickle
import h5py
import os
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
print(ls1)
dic1={}
import numpy as np
count=0
for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('color/'+i)
    if '.DS_Store' in ls2:
        ls2.remove('.DS_Store')
    for j in ls2:
        #im1=np.asarray(sm.imread('color/'+i+'/'+j))
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        count=count+1
print("Reach 1 \n")
ls1=os.listdir('color')
if '.DS_Store' in ls1:
    ls1.remove('.DS_Store')
dic1={}
X=np.zeros((count,256,256,3))
Y=np.zeros((count,1))
vap=0

for idx,i in enumerate(ls1):
    dic1[i]=idx
    ls2=os.listdir('color/'+i)
    if '.DS_Store' in ls2:
        ls2.remove('.DS_Store')
    for idx2,j in enumerate(ls2):
        print(str(idx) + " " + i + " " + str(idx2) + " " + j)
        X[vap, :, :, :]=cv2.imread('color/'+i+'/'+j)
        Y[vap,0]=idx
        #temp=np.zeros((len(im1),len(im1[0]),len(im1[0][0])   ))
        vap=vap+1

print(X.shape)
h5f = h5py.File("variables.h5",'w')
h5f.create_dataset("Y",data = Y)
print("Y done")
h5f.create_dataset("X",data = X)
print("X done")

# with open("variables.pickle",'wb') as f:
#     print("Save Variables")
#     pickle.dump(Y,f,pickle.HIGHEST_PROTOCOL)
#     print("Y done")
#     pickle.dump(X,f,pickle.HIGHEST_PROTOCOL)
#     print("X done")
