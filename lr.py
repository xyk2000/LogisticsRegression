import numpy as np
np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticsReg():
    def __init__(self,learn_rate=1e-3,niter=100000):
        self.lr=learn_rate
        self.iter=int(niter)

    def fit(self,x,y):
        x=np.insert(x, 0, values=1, axis=1)
        w=np.random.uniform(size=(x.shape[1]))

        for i in range(self.iter):
            p = sigmoid(x.dot(w))
            w -= self.lr * x.T.dot(p - y)
            
        self.w=w

    def predict(self,x):
        x=np.insert(x, 0, values=1, axis=1)
        return sigmoid(x.dot(self.w))


if __name__=='__main__':
    data=np.loadtxt('data.csv',delimiter=',',skiprows=1)
    np.random.shuffle(data)

    x=data[:600,:-1]
    y=data[:600,-1]

    mean=np.mean(x,axis=0)
    std=np.std(x,axis=0)
    x=(x-mean)/std
    
    x_val=data[600:,:-1]
    y_val=data[600:,-1]
    x_val=(x_val-mean)/std

    model=LogisticsReg()
    model.fit(x,y)
    
    import sklearn.metrics
    pred=model.predict(x_val)
    print(sklearn.metrics.roc_auc_score(y_val,pred))
    '''
    with open('acc_thres.csv','w') as f:
        for i in range(100):
            pred=model.predict(x_val)>(i/100)
            acc=sklearn.metrics.accuracy_score(y_val,pred)
            f.write('%s,%s\n'%(i/100,acc))
    '''
    with open('acc_lr.csv','w') as f:
        for i in range(-10,1):
            model=LogisticsReg(10**i)
            model.fit(x,y)
            pred=model.predict(x_val)>0.5
            acc=sklearn.metrics.accuracy_score(y_val,pred)
            f.write('%s,%s\n'%(10**i,acc))
