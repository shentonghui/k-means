import numpy as np                                  #引入numpy模块别名np
import numpy.random as random                       #别名random
import matplotlib.pyplot as plt                     #引入模块,别名为plt
import time                                         #引入time模块

#random.seed()                                      #设置随机数种子

def ishow(C0,C1,C2,mu):                             #作图函数
    c0 = '(' + str(round(mu[0,0],2)) + \
         ',' + str(round(mu[0,1],2)) + ')'          #需要显示的各中心点坐标
    c1 = '(' + str(round(mu[1,0],2)) + \
         ',' + str(round(mu[1,1],2)) + ')'
    c2 = '(' + str(round(mu[2,0],2)) + \
         ',' + str(round(mu[2,1],2)) + ')'
    plt.plot(*C0.T,'r.')                            #C0簇用红色点标记
    plt.plot(*C1.T,'g.')                            #C1簇用绿色点标记
    plt.plot(*C2.T,'b.')                            #C2簇用蓝色点标记
    plt.plot(*mu[0],'ro',MarkerSize=10,label = c0)  #C0簇中心红色实心圆标记
    plt.plot(*mu[1],'go',MarkerSize=10,label = c1)  #C1簇中心绿色实心圆标记
    plt.plot(*mu[2],'bo',MarkerSize=10,label = c2)  #C2簇中心蓝色实心圆标记
    plt.axis([0,2.5,0,2.5])                         #坐标轴范围
    plt.legend()                                    #图注
    plt.xlabel('X')                                 #X轴标签
    plt.ylabel('Y')                                 #Y轴标签
    plt.title('K-means')                            #图表标题
    plt.show()                                      #显示图表

def Create(x,y,num):                                #生成中心为(x,y)的随机点
    s = np.zeros((num,2))                           #用于存储符合条件的随机点
    mu = np.array([[x,y]])                          #均值
    Sigma = np.array([[0.06,0.01],[0.01,0.03]])     #标准差
    t=random.multivariate_normal(mu[0],Sigma,num*100)#生成高斯分布随机点   
    i = 0
    k = 0
    while k<num and i<num*100:                      #查找符合条件的点
        if(0<=t[i,0]<=2.5 and 0<=t[i,1]<=2.5):      #查找符合范围的随机点
            s[k,:]=t[i,:]                           #将符合范围的随机点加入s
            k += 1
        i += 1
    return s

def k_means(s):
    k = 3                                           #聚类簇数
    
    #随机选择k个样本作为初始均值向量
    mu = np.zeros((k,2))
    temp = random.choice(range(sum),k,replace=False)#无放回的随机采样k个整数
    temp.sort()
    for i in range(3):
        mu[i,:] = s[temp[i],:]                      #存储初始均值向量
    mu2 = np.zeros((k,2))                           #存储迭代后的均值向量
    
    change = 1                                      #标记均值向量是否发生变化
    count =0                                        #记录迭代次数
    while(change):                                  #均值向量发生变化时循环
        C0 = np.array([[0.0,0.0]])                  #迭代后的簇划分
        C1 = np.array([[0.0,0.0]])
        C2 = np.array([[0.0,0.0]])
        dis = np.zeros((sum,k))                     #存储各样本与均值向量的距离
        for j in range(sum):
            for i in range(k):
                dis[j,i]=np.sqrt((s[j,0]-mu[i,0])**2+
                                (s[j,1]-mu[i,1])**2)#计算距离
            min_flag = np.argmin(dis[j,:])          #确定簇标记
            if min_flag == 0:                       #将样本划入相应的簇中
                if((C0==np.zeros((1,2))).all()):    #判断簇中是否为空
                    C0[0,:] = s[j,:]                #簇中添入首个样本
                else:
                    C0 = np.row_stack((C0,s[j,:]))  #将样本添加入簇
            elif min_flag ==1:                      
                if((C1==np.zeros((1,2))).all()):
                    C1[0,:] = s[j,:]
                else:
                    C1 = np.row_stack((C1,s[j,:]))
            else:
                if((C2==np.zeros((1,2))).all()):
                    C2[0,:] = s[j,:]
                else:
                    C2 = np.row_stack((C2,s[j,:]))

        #从各簇中分别求出新的均值向量
        mu2[0,:] = np.mean(C0,axis=0)             
        mu2[1,:] = np.mean(C1,axis=0)
        mu2[2,:] = np.mean(C2,axis=0)
        if(not((mu==mu2).all())):                   #如果均值向量有变化
            mu = np.copy(mu2)                       #更新均值向量
        else:
            change = 0                              #更改change标记的值   
        count += 1                                  #迭代次数加1
        if(count%5==0):                             #每隔5次迭代画出效果图
            ishow(C0,C1,C2,mu)                      #作图函数
                                 
    print("总共迭代次数:%d"%count)                    #总迭代次数
    print("各类中心点坐标：(%4.2f,%4.2f)(%4.2f,%4.2f)(%4.2f,%4.2f)"
          %(mu[0,0],mu[0,1],mu[1,0],
            mu[1,1],mu[2,0],mu[2,1]))               #输出最终中心点坐标
    ishow(C0,C1,C2,mu)                              #最终迭代结果
    
    
start = time.clock()                                #记录程序运行开始时间
sum = 600                                           #总点数
s = np.vstack((Create(1.0,1.0,200),
               Create(1.0,2.0,200),
               Create(2.0,2.0,200)))                #生成随机点
k_means(s)                                          #k-means聚类算法
end = time.clock()                                  #记录程序运行结束时间
print ("Running time:%f s" %(end - start))          #输出运行所用时间

