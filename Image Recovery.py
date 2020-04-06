import cv2
import numpy as np
import math
import random
import sklearn
import scipy.misc
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy
import PIL
import imageio
import copy
img = cv2.imread('2.bmp')
img2 = cv2.imread('2.bmp')
# cv2.cvtColor(img,img,CV_BGR2GRAY)
# print(img.shape)
cv2.imshow('lenna', img)

img1 = img[:, :, 1]
def corruptPic(img,num):
    t=img
    row=img.shape[0]
    col=img.shape[1]
    for i in range(num):
        x=random.randint(0,row-1)
        y=random.randint(0,col-1)
        t[x][y]=0
    return t

print(img2.shape)
print(img2.dtype)
# print(imgtmp[0].shape)

print((np.array(img1)).shape)


# define the function for spliting the image array into small pieces
def split(image, n):
    data = np.split(image, image.shape[0] / n, axis=0)
    res = []
    for arr in data:
        res.extend(np.split(arr, arr.shape[1] / n, axis=1))
    '''
    ans=[]
    for i in range(len(res)):
        tmp = []
        for p in range(n):
            for q in range(n):
                tmp.append(res[i][p][q])
        ans.append(tmp)
        '''

    return res


def matrixToVector(matrix):
    ans = []
    row, col = (np.array(matrix)).shape[0], (np.array(matrix)).shape[1]
    for i in range(row):
        for j in range(col):
            ans.append(matrix[i][j])
    return ans


def getxy(index, wid):
    x = index / wid
    y = index % wid
    return int(x), int(y)


'''
def DCT(matrix):
    row,col=(np.array(matrix)).shape[0],(np.array(matrix)).shape[1]
    ans=[]
    width=int(col**0.5)
    for n in range(row):
        tmp=[]
        for index in range(col):
            x,y=getxy(index,width)
            sum=0
            for p in range(1,width+1):
                for q in range(1,width+1):
                    if(p==1 and q==1):
                        sum=(1/width)*math.cos(math.pi*(2*x-1)*(p-1)/(2*width))*math.cos(math.pi*(2*y-1)*(q-1)/(2*width))
                    elif(p==1 or q==1):
                        sum=math.sqrt(2)*(1/width)*math.cos(math.pi*(2*x-1)*(p-1)/(2*width))*math.cos(math.pi*(2*y-1)*(q-1)/(2*width))
                    else:
                        sum = (2 / width) * math.cos(math.pi * (2 * x - 1) * (p - 1) / (2 * width)) * math.cos(math.pi * (2 * y - 1) * (q - 1) / (2 * width))
                    tmp.append(sum)
            ans.append(tmp)
    return ans
    '''

#caluculate the transform matrix
def DCT(matrix):
    row, col = (np.array(matrix[0])).shape[0], (np.array(matrix[0])).shape[1]
    ans = []
    length = len(matrix)
    for i in range(length):
        tmp2 = []
        for x in range(1,row+1):
            for y in range(1,col+1):
                tmp = []
                for p in range(1,row+1):
                    for q in range(1,col+1):
                        if (p == 1 and q == 1):
                            sum = (1 / row) * math.cos(math.pi * (2 * x - 1) * (p - 1) / (2 * row)) * math.cos(
                                math.pi * (2 * y - 1) * (q - 1) / (2 * row))
                        elif (p == 1 or q == 1):
                            sum = math.sqrt(2) * (1 / row) * math.cos(
                                math.pi * (2 * x - 1) * (p - 1) / (2 * row)) * math.cos(
                                math.pi * (2 * y - 1) * (q - 1) / (2 * row))
                        else:
                            sum = (2 / row) * math.cos(math.pi * (2 * x - 1) * (p - 1) / (2 * row)) * math.cos(
                                math.pi * (2 * y - 1) * (q - 1) / (2 * row))
                        tmp.append(sum)
                tmp2.append(tmp)
        ans.append(tmp2)
    return ans


def reshapeMatrix(matrix):
    ans = []
    for i in range(len(matrix)):
        tmp = []
        for j in range(len(matrix[i])):
            for q in matrix[i][j]:
                tmp+=q
        ans.append(tmp)
    return ans


# %%
p = split(img1, 8)
print(p[0].shape)
#%%
def toVector(res):
    tmp=[]
    for i in range(len(res)):
        for j in range(len(res[i])):
            if (j == 0):
                a = res[i][j]
            elif (j == len(res[0]) - 1):
                a = np.hstack((a, res[i][j]))
                tmp.append(a)
            else:
                a = np.hstack((a, res[i][j]))
    return tmp

#%%
ans = DCT(p)
# ans=reshapeMatrix(ans)
#p=reshapeMatrix(p)


print('image:', (np.array(p)).shape)
print(len(p))

print(p[0])

print('trans:', (np.array(ans)).shape)
print(len(ans))
print(ans[0])

# define the function for spliting the image array into small pieces


# %%
'''
#q=reshapeMatrix(p)
q=np.reshape(p,(len(p),64))
print('reshape:', (np.array(q)).shape)
print('p', p[0][0][2])
'''
q=toVector(p)
#%%
def sampleNumber(n,length):
    ans=[]
    sum=0
    while(sum!=n):
        r = np.random.randint(length)
        if r not in ans:
            ans.append(r)
            sum+=1
    return ans
def sampleMatrix(ans,matrix):
    s=10
    length=len(ans)
    n=len(ans[0])
    x=[]
    y=[]
    for i in range(length):
        idx=sampleNumber(s,n)
        xsample=ans[i][idx]
        ysample=matrix[i][idx]
        x.append(xsample)
        y.append(ysample)
    return x,y
#xtrain,ytrain=sampleNumber(q.tolist(),ans)

index=sampleNumber(20,64)
#sampleindex=sampleNumber(s,len(q[0]),len(q))
#%%
def getSamplematrix(q,ans,sampleindex):
    length=len(q)
    x=[]
    y=[]
    for i in range(length):
        tmp=q[i][sampleindex[i]]
        x.append(tmp)
        tmp=ans[i][sampleindex[i]]
        y.append(tmp)
    return x,y
#xdata,ydata=getSamplematrix(q,np.array(ans),sampleindex)
#test=ans[0][sampleindex[0]]


#%%
def Mse(predict, price):
    sum = 0
    for i in range(len(predict)):
        sum += (predict[i] - price[i]) ** 2
    return sum / len(price)
def Nfold(x,y,M,alp):
    kf = model_selection.ShuffleSplit(n_splits=M,test_size=(int)(len(x)/6),random_state=0)
    model=sklearn.linear_model.Lasso(alpha=alp)
    ans=[]
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        model.fit(X_train,Y_train)
        tmp=model.predict(X_test)
        ans.append(Mse(tmp,Y_test))
    return np.mean(ans),model.coef_

'''
ans=np.array(ans)
xdata1=ans[0][index,:]
ydata1=q[0][index]
'''
#%%
print(ydata1)
model=sklearn.linear_model.LinearRegression()
model.fit(xdata1,ydata1.ravel())
ppp=model.coef_
sdsdssdsds=ans[0].dot(np.mat(model.coef_).T)
sddd=np.dot(np.mat(ans[0]),np.mat(model.coef_).T)
print(sddd)
'''
print(ans[0])
print(model.predict(xdata1))
'''
#%%
alpha=[0.0000001,0.00001,0.001,0.01,0.1,0.2,0.3,0.5,0.7,0.8]
qwewq,para=Nfold(xdata1,ydata1,20,0.1)
ehus=ans[0].dot(para)
#%%

def imgRecover(q,ans,SampleNumber,imageSize):
    blockPara=[]
    corrPic=copy.copy(q)
    alpha = [ 0.00001, 0.001, 0.01, 0.1, 0.5 ]
    for i in range(len(q)):
        index = sampleNumber(SampleNumber, imageSize)
        for k in range(imageSize):
            if k not in index:
                corrPic[i][k]=0
        xdata=ans[i][index]
        ydata=q[i][index]
        #store the lmdavalue for each block
        dict={}
        for alp in alpha:
            t,para=Nfold(xdata, ydata, 20, alp)
            dict[alp]=t
        s=sorted(dict.items(), key=lambda item: item[1])
        print('best alpha:',s[0][0])
        model=sklearn.linear_model.Lasso(alpha=s[0][0])
        model.fit(xdata,ydata)
        para=model.predict(ans[i])
        blockPara.append(para)
    return blockPara,corrPic
#recover,c=imgRecover(q,ans,20,64)

#%%
print(img1.shape)




'''
def getPixel(ans,para):
    answer=[]
    for i in range(len(ans)):
        c=ans[i].dot(para[i])
        answer.append(c)
    return answer
restorePixel=getPixel(ans,blockPara)
'''
#tesh=np.reshape(restorePixel[0],(8,8))

#%%
def getPicmatrix(matrix, wid):
    finalpic=[]
    width=192
    for i in matrix:
        a=np.reshape(i,(wid,wid))
        finalpic.append(a)
    return finalpic
def recoverPic(wid,matrix):
    b=8
    n=wid/8

    ans=[]
    for i in range(len(matrix)):
        if(i%n==0):
            a=matrix[i]
        elif((i+1)%n==0):
            a = np.hstack((a, matrix[i]))
            ans.append(a)

        else:
            a = np.hstack((a, matrix[i]))
    for i in range(len(ans)):
        if(i==0):
            a=ans[i]
        else:
            a=np.vstack((a,ans[i]))
    return a
'''
final33=getPicmatrix(recover,8)
#final2=np.reshape(final33,(200,192))
final22=recoverPic(192,final33)
plt.imshow(final22,cmap='gray')
plt.show()
print(img1)
'''
#%%
def MSE(matrix1,matrix2):
    row=len(matrix1)
    col=len(matrix1[0])
    sum=[]
    for i in range(row):
        for j in range(col):
            sum.append(pow((matrix1[i][j]-matrix2[i][j]),2))
    return np.mean(sum)
l=MSE(img1,img1)
#%%
#M--sample time  n--64  num--sample number 20/30/50
def minLamda(M,alpha,x,y,n,num):
    nfold=6
    ans= dict.fromkeys(alpha, 0)
    for i in range(M):
        index = sampleNumber(n,num)
        xdata=x[index]
        ydata=y[index]
        for alp in alpha:
            tmp= kfold(xdata, ydata, nfold, alp)
            ans[alp]+=tmp
    return ans
alpha=[0.1,0.5]
test1=minLamda(5,alpha,xdata,ydata,64,20)
# q pixcel  matrix--transformmatrix  num--samplenumber
def calApha(q,matrix,num,alpha):
    length=len(q)
    #sample time
    M=2
    #kfold number
    nfold=6
    n=len(q[0])
    ans=[]
    for i in range(length):
        for i in range(M):
            index=sampleNumber(n, num)
            x=q[i][index]
            y=matrix[i][index]
            for alp in alpha:
                tmp=kfold(x, y, nfold, alp)
                ans.append(tmp)
    return ans

#alpha=[0.1]
#chenxi=calApha(q,ans,20,alpha)

#%%


#this is fishboat processing

#get the corrupted picture
#imgcor=corruptPic(img1,10000)
#plt.imshow(imgcor)
#plt.title("corrupted image")
#plt.show()
#%%
fishboatCorruptimage= split(img1, 8)
fishboatTransMatrix = DCT(fishboatCorruptimage)
fishboatCorruptimage=toVector(fishboatCorruptimage)
#%%

#strating from here sample 10
fishboatTransMatrix=np.array(fishboatTransMatrix)
recoverFishboat,corrpImg=imgRecover(fishboatCorruptimage,fishboatTransMatrix,10,64)
#%%
fishboatRec=getPicmatrix(recoverFishboat,8)
imgCor=getPicmatrix(corrpImg,8)
#final2=np.reshape(final33,(200,192))
imgCor=recoverPic(192,imgCor)
fishboatImageRec=recoverPic(192,fishboatRec)

#plt.imshow(fishboatImageRec)
#plt.show()

#%%
#plt.show()
fishboatRec=ndimage.median_filter(fishboatImageRec, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 10 Recover image')
ax[0][1].imshow(fishboatImageRec,cmap='gray')
ax[0][0].title.set_text('Sample 10 Corrupted image')
ax[0][0].imshow(imgCor,cmap='gray')
ax[1][0].title.set_text('Sample 10 Recover Median filter image')
ax[1][0].imshow(fishboatRec,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%

#starting from here it is sample 20
mseF10=MSE(img1,fishboatImageRec)
mseF102=MSE(img1,fishboatRec)
imageio.imwrite('fishboat10.bmp', fishboatRec.astype(np.uint8))
#%%
fishboatTransMatrix2=np.array(fishboatTransMatrix)
recoverFishboat2,corrpImg2=imgRecover(fishboatCorruptimage,fishboatTransMatrix2,20,64)
#%%
fishboatRec2=getPicmatrix(recoverFishboat2,8)
imgCor2=getPicmatrix(corrpImg2,8)
#final2=np.reshape(final33,(200,192))
imgCor2=recoverPic(192,imgCor2)
fishboatImageRec2=recoverPic(192,fishboatRec2)

#plt.imshow(fishboatImageRec)
#plt.show()

#%%
#plt.show()
fishboatRec2=ndimage.median_filter(fishboatImageRec2, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 20 Recover image')
ax[0][1].imshow(fishboatImageRec2,cmap='gray')
ax[0][0].title.set_text('Sample 20 Corrupted image')
ax[0][0].imshow(imgCor2,cmap='gray')
ax[1][0].title.set_text('Sample 20 Recover Median filter image')
ax[1][0].imshow(fishboatRec2,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseArray=[]
mseArray.append(mseF10)
mseF20=MSE(img1,fishboatImageRec2)
mseF202=MSE(img1,fishboatRec2)
#imageio.imwrite('fishboat20.bmp', fishboatRec2.astype(np.uint8))
mseArray.append(mseF20)
#%%

#%%
fishboatTransMatrix2=np.array(fishboatTransMatrix)
recoverFishboat3,corrpImg3=imgRecover(fishboatCorruptimage,fishboatTransMatrix2,30,64)
#%%
fishboatRec3=getPicmatrix(recoverFishboat3,8)
imgCor3=getPicmatrix(corrpImg3,8)
#final2=np.reshape(final33,(200,192))
imgCor3=recoverPic(192,imgCor3)
fishboatImageRec3=recoverPic(192,fishboatRec3)

#plt.imshow(fishboatImageRec)
#plt.show()

#%%
#plt.show()
fishboatRec3=ndimage.median_filter(fishboatImageRec3, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 30 Recover image')
ax[0][1].imshow(fishboatImageRec3,cmap='gray')
ax[0][0].title.set_text('Sample 30 Corrupted image')
ax[0][0].imshow(imgCor3,cmap='gray')
ax[1][0].title.set_text('Sample 30 Recover Median filter image')
ax[1][0].imshow(fishboatRec3,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseF30=MSE(img1,fishboatImageRec3)
mseF302=MSE(img1,fishboatRec3)
#imageio.imwrite('fishboat30.bmp', fishboatRec3.astype(np.uint8))
mseArray.append(mseF30)
#%%
mseArray=[]
mseArray.append(mseF10)
mseArray.append(mseF20)
mseArray.append(mseF30)
mseArray.append(mseF40)
#%%
fishboatTransMatrix2=np.array(fishboatTransMatrix)
recoverFishboat4,corrpImg4=imgRecover(fishboatCorruptimage,fishboatTransMatrix2,40,64)
#%%
fishboatRec4=getPicmatrix(recoverFishboat4,8)
imgCor4=getPicmatrix(corrpImg4,8)
#final2=np.reshape(final33,(200,192))
imgCor4=recoverPic(192,imgCor4)
fishboatImageRec4=recoverPic(192,fishboatRec4)

#plt.imshow(fishboatImageRec)
#plt.show()

#%%
#plt.show()
fishboatRec4=ndimage.median_filter(fishboatImageRec4, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 40 Recover image')
ax[0][1].imshow(fishboatImageRec4,cmap='gray')
ax[0][0].title.set_text('Sample 40 Corrupted image')
ax[0][0].imshow(imgCor4,cmap='gray')
ax[1][0].title.set_text('Sample 40 Recover Median filter image')
ax[1][0].imshow(fishboatRec4,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseF40=MSE(img1,fishboatImageRec4)
mseF402=MSE(img1,fishboatRec4)
imageio.imwrite('fishboat40.bmp', fishboatRec4.astype(np.uint8))

#%%
fishboatCorruptimage= split(img1, 8)
fishboatTransMatrix = DCT(fishboatCorruptimage)
fishboatCorruptimage=toVector(fishboatCorruptimage)
#%%
fishboatTransMatrix5=np.array(fishboatTransMatrix)
recoverFishboat5,corrpImg5=imgRecover(fishboatCorruptimage,fishboatTransMatrix5,50,64)
#%%
fishboatRec5=getPicmatrix(recoverFishboat5,8)
imgCor5=getPicmatrix(corrpImg5,8)
#final2=np.reshape(final33,(200,192))
imgCor5=recoverPic(192,imgCor5)
fishboatImageRec5=recoverPic(192,fishboatRec5)

#plt.imshow(fishboatImageRec)
#plt.show()

#%%
#plt.show()
fishboatRec5=ndimage.median_filter(fishboatImageRec5, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 50 Recover image')
ax[0][1].imshow(fishboatImageRec5,cmap='gray')
ax[0][0].title.set_text('Sample 50 Corrupted image')
ax[0][0].imshow(imgCor5,cmap='gray')
ax[1][0].title.set_text('Sample 50 Recover Median filter image')
ax[1][0].imshow(fishboatRec5,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseF50=MSE(img1,fishboatImageRec5)
mseF502=MSE(img1,fishboatRec5)
imageio.imwrite('fishboat50.bmp', fishboatRec5.astype(np.uint8))
#%%
mseArray=[]
mseArray.append(mseF10)
mseArray.append(mseF20)
mseArray.append(mseF30)
mseArray.append(mseF40)
mseArray.append(mseF50)
#%%
x=[10,20,30,40,50]

name_list = ['10 sample','20 sample','30 sample','40 sample','50 sample']
plt.bar(x,mseArray,tick_label=name_list,width=7)
plt.title('MSE for fishboat picture of different sample')
plt.show()
#%%
imgLenna = cv2.imread('1.bmp')
imgLenna=imgLenna[:, :, 1]
lennaCorruptimage= split(imgLenna, 16)
lennaTransMatrix = DCT(lennaCorruptimage)
lennaCorruptimage=toVector(lennaCorruptimage)
#%%
lennaTransMatrix1=np.array(lennaTransMatrix)
recoverLenna,corrpImg1=imgRecover(lennaCorruptimage,lennaTransMatrix1,10,256)
#%%
def getPicmatrix(matrix, wid):
    finalpic=[]
    width=192
    for i in matrix:
        a=np.reshape(i,(wid,wid))
        finalpic.append(a)
    return finalpic
def recoverPic(wid,matrix):
    b=8
    n=wid/16

    ans=[]
    for i in range(len(matrix)):
        if(i%n==0):
            a=matrix[i]
        elif((i+1)%n==0):
            a = np.hstack((a, matrix[i]))
            ans.append(a)

        else:
            a = np.hstack((a, matrix[i]))
    for i in range(len(ans)):
        if(i==0):
            a=ans[i]
        else:
            a=np.vstack((a,ans[i]))
    return a
#%%
lennaRec1=getPicmatrix(recoverLenna,16)
imgCor1=getPicmatrix(corrpImg1,16)
#final2=np.reshape(final33,(200,192))
imgCor1=recoverPic(512,imgCor1)
lennaImageRec1=recoverPic(512,lennaRec1)
#%%
print(len(lennaRec1))
#%%
#plt.show()
lennaRec1=ndimage.median_filter(lennaImageRec1, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 10 Recover image')
ax[0][1].imshow(lennaImageRec1,cmap='gray')
ax[0][0].title.set_text('Sample 10 Corrupted image')
ax[0][0].imshow(imgCor1,cmap='gray')
ax[1][0].title.set_text('Sample 10 Recover Median filter image')
ax[1][0].imshow(lennaRec1,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseL10=MSE(imgLenna ,lennaRec1)
mseL102=MSE(imgLenna ,lennaImageRec1)
imageio.imwrite('Lenna10.bmp', lennaImageRec1.astype(np.uint8))
#%%
imgLenna = cv2.imread('1.bmp')
imgLenna=imgLenna[:, :, 1]
#%%
lennaCorruptimage= split(imgLenna, 16)
lennaTransMatrix = DCT(lennaCorruptimage)
lennaCorruptimage=toVector(lennaCorruptimage)
#%%
lennaTransMatrix2=np.array(lennaTransMatrix)
recoverLenna2,corrpImg2=imgRecover(lennaCorruptimage,lennaTransMatrix2,30,256)
#%%
lennaRec2=getPicmatrix(recoverLenna2,16)
imgCor2=getPicmatrix(corrpImg2,16)
#final2=np.reshape(final33,(200,192))
imgCor2=recoverPic(512,imgCor2)
lennaImageRec2=recoverPic(512,lennaRec2)
#%%
#plt.show()
lennaRec2=ndimage.median_filter(lennaImageRec2, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 30 Recover image')
ax[0][1].imshow(lennaImageRec2,cmap='gray')
ax[0][0].title.set_text('Sample 30 Corrupted image')
ax[0][0].imshow(imgCor2,cmap='gray')
ax[1][0].title.set_text('Sample 30 Recover Median filter image')
ax[1][0].imshow(lennaRec2,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseL20=MSE(imgLenna ,lennaRec2)
mseL202=MSE(imgLenna ,lennaImageRec2)
imageio.imwrite('Lenna20.bmp', lennaImageRec2.astype(np.uint8))
#%%
lennaCorruptimage= split(imgLenna, 16)
lennaCorruptimage=toVector(lennaCorruptimage)
#%%
recoverLenna2,corrpImg2=imgRecover(lennaCorruptimage,lennaTransMatrix2,50,256)
#%%
lennaRec2=getPicmatrix(recoverLenna2,16)
imgCor2=getPicmatrix(corrpImg2,16)
#final2=np.reshape(final33,(200,192))
imgCor2=recoverPic(512,imgCor2)
lennaImageRec2=recoverPic(512,lennaRec2)
#%%
#plt.show()
lennaRec2=ndimage.median_filter(lennaImageRec2, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 50 Recover image')
ax[0][1].imshow(lennaImageRec2,cmap='gray')
ax[0][0].title.set_text('Sample 50 Corrupted image')
ax[0][0].imshow(imgCor2,cmap='gray')
ax[1][0].title.set_text('Sample 50 Recover Median filter image')
ax[1][0].imshow(lennaRec2,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseL30=MSE(imgLenna ,lennaRec2)
mseL302=MSE(imgLenna ,lennaImageRec2)
imageio.imwrite('Lenna50.bmp', lennaImageRec2.astype(np.uint8))
#%%
lennaCorruptimage= split(imgLenna, 16)
lennaCorruptimage=toVector(lennaCorruptimage)
#%%
recoverLenna2,corrpImg2=imgRecover(lennaCorruptimage,lennaTransMatrix2,100,256)
#%%
lennaRec2=getPicmatrix(recoverLenna2,16)
imgCor2=getPicmatrix(corrpImg2,16)
#final2=np.reshape(final33,(200,192))
imgCor2=recoverPic(512,imgCor2)
lennaImageRec2=recoverPic(512,lennaRec2)
#%%
#plt.show()
lennaRec2=ndimage.median_filter(lennaImageRec2, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 100 Recover image')
ax[0][1].imshow(lennaImageRec2,cmap='gray')
ax[0][0].title.set_text('Sample 100 Corrupted image')
ax[0][0].imshow(imgCor2,cmap='gray')
ax[1][0].title.set_text('Sample 100 Recover Median filter image')
ax[1][0].imshow(lennaRec2,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseL40=MSE(imgLenna ,lennaRec2)
mseL402=MSE(imgLenna ,lennaImageRec2)
imageio.imwrite('Lenna100.bmp', lennaImageRec2.astype(np.uint8))
#%%
lennaCorruptimage= split(imgLenna, 16)
lennaCorruptimage=toVector(lennaCorruptimage)
#%%
recoverLenna2,corrpImg2=imgRecover(lennaCorruptimage,lennaTransMatrix2,150,256)
#%%
lennaRec2=getPicmatrix(recoverLenna2,16)
imgCor2=getPicmatrix(corrpImg2,16)
#final2=np.reshape(final33,(200,192))
imgCor2=recoverPic(512,imgCor2)
lennaImageRec2=recoverPic(512,lennaRec2)
#%%
#plt.show()
lennaRec2=ndimage.median_filter(lennaImageRec2, size=(3, 3))
f, ax = plt.subplots(2, 2, sharey=True)
ax[0][1].title.set_text('Sample 150 Recover image')
ax[0][1].imshow(lennaImageRec2,cmap='gray')
ax[0][0].title.set_text('Sample 150 Corrupted image')
ax[0][0].imshow(imgCor2,cmap='gray')
ax[1][0].title.set_text('Sample 150 Recover Median filter image')
ax[1][0].imshow(lennaRec2,cmap='gray')
ax[-1, -1].axis('off')
plt.show()
#%%
mseL50=MSE(imgLenna ,lennaRec2)
mseL502=MSE(imgLenna ,lennaImageRec2)
imageio.imwrite('Lenna150.bmp', lennaImageRec2.astype(np.uint8))
#%%
mseArrayl=[]
mseArrayl.append(mseL10)
mseArrayl.append(mseL20)
mseArrayl.append(mseL30)
mseArrayl.append(mseL40)
mseArrayl.append(mseL50)
x=[10,30,50,100,150]

name_list = ['10 sample','30 sample','50 sample','100 sample','150 sample']
plt.bar(x,mseArrayl,tick_label=name_list,width=7)
plt.title('MSE for lenna picture of different sample')
plt.show()