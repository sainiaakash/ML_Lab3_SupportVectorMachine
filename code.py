
# check the following link for using cvxopt qp solver
# http://cvxopt.org/examples/tutorial/qp.html

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
import matplotlib.pyplot as plt
             
class SVM(object):

    def __init__(self, kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)
            
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Kernel/Gram matrix
        # to do - compute the kernel matrix given the choice of the kernel function
        gram = np.ones((n_samples, n_samples))
        i = 0
        j = 0
        
        while i < n_samples:
            j=0
            while j < n_samples:
                gram[i,j] = self.kernel(X[i], X[j])
                j+=1
            i+=1
        
        temp = np.ones(n_samples) * -1
        P = cvxopt.matrix(np.outer(y,y)*gram)
        q = cvxopt.matrix(temp)
        b = cvxopt.matrix(0.0)
        A = cvxopt.matrix(y, (1,n_samples))
        # Compute the parameters to be sent to the solver
        # P, q, A, b - refer to lab for more information.
        # remember to use cvxopt.matrix for storing the vector and matrices. 
        # to do
        if self.C is None: # linear separable case
            # Compute the parameters to be sent to the solver
            # G, h - refer to the lab for more information.
            # remember to use cvxopt.matrix for storing the vector and matrices. 
            # cvxopt does not work with numpy matrix
            # to do
            diag = np.diag(temp)
            G = cvxopt.matrix(diag)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # soft margin case
            # Compute the parameters to be sent to the solver
            # G, h - refer to the lab for information about it.
            # remember to use cvxopt.matrix for storing the vector and matrices. 
            # cvxopt does not work with numpy matrix
            # to do
            tmp1 = np.diag(temp)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples),np.ones(n_samples) * self.C)))
        # solve QP problem once we have all the parameters of the solver specifed, let us solve it! 
        # uncomment the line below once you have specified all the parameters.
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Obtain the Lagrange multipliers from the solution.
        alpha = np.ravel(solution['x'])
        # Support vectors have non zero Lagrange multipliers
        # apply a threshold on the value of alpha and identify the support vectors
        # print the fraction of support vectors.
        # to do
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Weight vector
        # compute the weight vector using the support vectors only when using linear kernel
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            # to do
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None
            
        # Intercept
        # computer intercept term by taking the average across all support vectors
        self.w0 = 0
        # to do
        n = 0
        while n < len(self.alpha):
            self.w0 += self.sv_y[n]
            self.w0 -= np.sum(self.alpha * self.sv_y * gram[ind[n],sv])
            n+=1
        self.w0 /= len(self.alpha)
        
    def transform(self, X): #transforming the data from lower dimension to higher dimension
        if self.w is not None:
            return np.dot(X,self.w) + self.w0
        else:
            pred = np.zeros(len(X))
            i = 0
            while i<len(X):
                x = 0
                for alpha, svy, sv in zip(self.alpha, self.sv_y, self.sv):
                    t1 = self.kernel(X[i], sv)
                    t2 = alpha *svy *t1
                    x += t2
                pred[i] = x
                i +=1
        return self.w0+pred 

    def predict(self, X):
        # implement the function to predict the class label for a test set.
        # return the class label and the output f(x) for a test data point
        # to do
        if self.w is not None:
            temp = np.dot(X, self.w) + self.w0
        else:
            pred = np.zeros(len(X))
            for i in range(len(X)):
                x = 0
                for alpha, svy, sv in zip(self.alpha, self.sv_y, self.sv):
                    t1 = self.kernel(X[i], sv)
                    t2 = alpha *svy *t1
                    x += t2
                pred[i] = x
            temp = pred + self.w0
        return np.sign(temp)
    
if __name__ == "__main__":
    
    def linear_kernel(x1, x2):
        # implement the linear kernel
        # to do
        dot = np.dot(x1,x2)
        return dot

    def polynomial_kernel(x1, x2,q=3):
        # implement the polynomial kernel
        # to do
        a = np.dot(x1,x2)
        b = 1+a
        return pow(b,q)

    def gaussian_kernel(x1,x2, s=5):
        # implement the radial basis function kernel
        # to do
        a = pow(linalg.norm(x1-x2),2)
        b = 2*pow(s,2)
        return np.exp(-a/b)
    

    def lin_separable_data():
        # generate linearly separable 2D data
        # remember to assign class labels as well.    
        # to do
        x1 = np.random.multivariate_normal([0,2], [[0.8, 0.6], [0.6, 0.8]],100)
        y1 = np.ones(len(x1))
        x2 = np.random.multivariate_normal([2,0], [[0.8, 0.6], [0.6, 0.8]],100)
        y2 = np.ones(len(x2)) * -1
        return x1, y1, x2, y2

    def circular_data():
        # let us complicate things a little to study the advantage of using Kernel functions
        # generate data that is separable using a circle
        # to do
        r = 20 * np.random.random(200)
        far = r > 10
        r[~far] *= 1.1
        r[far]  *= 1.2
        
        x = np.empty((200, 2))
        x[:, 0] = r * np.cos(np.pi*2*np.random.random(200))
        x[:, 1] = r * np.sin(np.pi*2*np.random.random(200))
        class1 = []
        class2 = []
        labels = np.ones(200)
        labels[far] = -1
        for i in range(0,200):
            if labels[i] == 1.0:
                class1.append(x[i])
            else:
                class2.append(x[i])
        y1 = np.ones(len(class1))
        y2 = np.ones(len(class2)) * -1
        return class1, y1, class2, y2
        

    def lin_separable_overlap_data():
        # for testing the soft margin implementation, 
        # generate linearly separable data where the instances of the two classes overlap.
        # to do
        x1 = np.random.multivariate_normal([0, 2], [[1.5, 1.0],[1.0,1.5]], 100)
        y1 = np.ones(len(x1))
        x2 = np.random.multivariate_normal([2, 0], [[1.5, 1.0],[1.0,1.5]], 100)
        y2 = np.ones(len(x2)) * -1
        return x1, y1, x2, y2

    def split_train_test(X1, y1, X2, y2):
        # split the data into train and test splits
        # to do
        fract = 90
        train_x  = np.vstack((X1[:fract], X2[:fract]))
        train_y  = np.hstack((y1[:fract], y2[:fract]))
        test_x   = np.vstack((X1[fract:], X2[fract:]))
        test_y   = np.hstack((y1[fract:], y2[fract:]))
        return train_x, train_y, test_x, test_y
        
    def plot(label,model,rep): #fitting the margin using the line equation
        a = (model.w)[0]
        b = model.w0
        c = (model.w)[1]
        d = label
        x = (-a*-2 - b + d)/c 
        y = (-a* 4 - b + d)/c
        pl.plot([-2,4],[x,y],rep)
        
    def plot_margin(X1_train, X2_train, model,name):
        # plot the margin boundaries (for the linear separable and overlapping case)
        # plot the data points
        # plot the w^Tx+w_0 = 1, w^Tx+w_0 = -1, and w^Tx+w_0 = 0, lines
        # highlight the support vectors.
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(model.sv[:,0], model.sv[:,1], s=250, c="g")
        plot(0,model,'-')
        plot(1,model,'--')
        plot(-1,model,'--')
        pl.axis("tight")
        pl.savefig(name+'.png')
        pl.show()
        

    def plot_contour(X1_train, X2_train, model,name):
        # plot the contours of the classifier
        # create a meshgrid and for every point in the grid compute the output f(x) of the classifier
        # use the classifier's output to plot the contours on the meshgrid
        pl.plot(X1_train[:,0], X1_train[:,1], "yo")
        pl.plot(X2_train[:,0], X2_train[:,1], "ro")
        pl.scatter(model.sv[:,0], model.sv[:,1], s=100, c="b")
        X1, X2 = np.meshgrid(np.linspace(-4,6,50), np.linspace(-4,6,50))
        X = np.vstack([X1.ravel(), X2.ravel()]).T
        Z = model.transform(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0],   colors='black', linewidths=1)
        pl.contour(X1, X2, Z + 1, [0.0],colors='green',linewidths=1)
        pl.contour(X1, X2, Z - 1, [0.0],colors='blue',linewidths=1)
        pl.legend(['+1','-1'],loc=2)
        pl.xlabel('x1')
        pl.ylabel('x2')
        pl.axis("tight")
        pl.savefig(name+'.png')
        pl.show()
        
        
    def accuracy(actual,predict):
        sum = 0
        for i in range(len(actual)):
            if actual[i]==predict[i]:
                sum+=1
        return (sum/len(actual))*100
        
    def linear_svm(k):
        # 1. generate linearly separable data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model (uses linear kernel)
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        X1,y1,X2,y2 = lin_separable_data()
        train_x, train_y,test_x,test_y = split_train_test(X1, y1, X2, y2)
        
        if k==1:   #linear-kernel model
            model = SVM(linear_kernel)
            model.fit(train_x, train_y)
            plot_margin(train_x[train_y==1], train_x[train_y==-1], model,'linear1')
            
            
        if k==2: #gaussian-kenel model
            model = SVM(gaussian_kernel)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1],model,'linear2')
            
            
        if k==3: #polynomial-kernel model
            model = SVM(polynomial_kernel)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1], model,'linear3')
            
        y_predict = model.predict(test_x)
        acc = accuracy(test_y,y_predict) #estimating the accuracy
        print('Accuracy:'+str(acc)+'%')


    def kernel_svm(k):
        # 1. generate non-linearly separable data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model using an appropriate kernel function
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the contours of the model's output using the plot_contour function
        
        x1, y1, x2, y2 = circular_data()
        train_x, train_y,test_x,test_y = split_train_test(x1, y1, x2, y2)
        
        if k==1: 
            model = SVM(linear_kernel)
            model.fit(train_x, train_y) #fitting the learned model
            plot_margin(train_x[train_y==1], train_x[train_y==-1], model,'kernel1')
            

        if k==2:
            model = SVM(gaussian_kernel)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1],model,'kernel2')
            
            
        if k==3:
            model = SVM(polynomial_kernel)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1], model,'kernel3')
            
            
        y_predict = model.predict(test_x)
        acc = accuracy(test_y,y_predict) #estimating the accuracy
        print('Accuracy:'+str(acc)+'%')
        
    def soft_svm(k):
        # 1. generate linearly separable overlapping data
        # 2. split the data into train and test sets
        # 3. create an SVM object called model (uses linear kernel, and the box penalty parameter)
        # 4. train the SVM using the fit function and the training data
        # 5. compute the classes of the model for the test data
        # 6. compute the accuracy of the model
        # 7. plot the training data points, and the margin.
        x1, y1, x2, y2 = lin_separable_overlap_data()
        train_x, train_y,test_x,test_y = split_train_test(x1, y1, x2, y2)
        
        if k==1:
            model = SVM(linear_kernel,C=1)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1],model,'soft1')
            
            
        if k==2:
            model = SVM(gaussian_kernel)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1],model,'soft2')
            
            
        if k==3:
            model = SVM(polynomial_kernel,C=1)
            model.fit(train_x, train_y)
            plot_contour(train_x[train_y==1], train_x[train_y==-1], model,'soft3')
            
            
        y_predict = model.predict(test_x)
        acc = accuracy(test_y,y_predict) #estimating the accuracy
        print('Accuracy:'+str(acc)+'%')
        
    # after you have implemented the kernel and fit functions let us test the implementations
    # uncomment each of the following lines as and when you have completed their implementations.
    
    
    ker_number = 1  #change this to use different kernel  
    linear_svm(ker_number)
    soft_svm(ker_number)
    kernel_svm(ker_number)
    





