import numpy as np

class Parameters:
    def __init__(self):
        pass
    
    def ForwardPropagation(self,X):
        W = self.W
        b = self.b
        A = [None]*len(W)
        H = [None]*len(b)
        for i in range(len(W)):
            if i == 0:
                A[i] = b[i] + W[i]@X
            else:
                A[i] = b[i] + W[i]@H[i-1]
            if i == len(W) - 1:
                H[i] = self.output_activation(A[i])
            else:
                H[i] = self.activation(A[i])
        self.A = A
        self.H = H
    
    def BackPropagation(self,X,Y):
        W = self.W
        b = self.b
        A = self.A
        H = self.H
        grad_a = [None]*len(A)
        grad_h = [None]*len(H)
        grad_w = [None]*len(W)
        grad_b = [None]*len(b)
        N = len(W)
        
        Y_hat = H[len(H)-1]
        grad_a[N-1] = Y_hat - Y
        for i in range(N-1,-1,-1):
            if i == 0:
                grad_w[i] = grad_a[i]@X.T
            else:
                grad_w[i] = grad_a[i]@H[i-1].T
            grad_b[i] = np.sum(grad_a[i],axis=1,keepdims=True)
            if i>0 :
                grad_h[i-1] = W[i].T@grad_a[i]
                grad_a[i-1] = grad_h[i-1]*self.der_activation(A[i-1])
        for i in range(N):
            W[i] += self.weight_decay*W[i]   
        self.grad_w = grad_w
        self.grad_b = grad_b
        self.W = W
 