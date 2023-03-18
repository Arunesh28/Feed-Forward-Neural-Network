import numpy as np

class sgd:
    def __init__(self):
        pass
    
    def update_sgd_params(self,eta):
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        W = self.params.W
        b = self.params.b
        N = len(W)
        for i in range(N):
            W[i] = W[i] - eta*grad_w[i]
            b[i] = b[i] - eta*grad_b[i]
        
class momentum():
    
    def __init__(self):
        pass
    
    def updade_momentum_params(self,eta,beta):
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        W = self.params.W
        b = self.params.b
        u_W = self.u_params.W
        u_b = self.u_params.b
        N = len(grad_w)
        for i in range(N):
            u_W[i] = beta*u_W[i] + grad_w[i]
            u_b[i] = beta*u_b[i] + grad_b[i]
            W[i] = W[i] - eta*u_W[i]
            b[i] = b[i] - eta*u_b[i]

class nesterov():
    
    def __init__(self):
        pass
    
    def update_nesterov_params(self,eta,beta,X,Y):
        W = self.params.W
        b = self.params.b
        u_W = self.u_params.W
        u_b = self.u_params.b
        g_W = self.lookahead_params.W
        g_b = self.lookahead_params.b
        N = len(W)
        
        self.params.ForwardPropagation(X)
        self.params.BackPropagation(X,Y)
        
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        
        for i in range(N):
            g_W[i] = grad_w[i]
            g_b[i] = grad_b[i]
        
        
        for i in range(N):
            u_W[i] = beta*u_W[i] + g_W[i]
            u_b[i] = beta*u_b[i] + g_b[i]
            W[i] = W[i] - eta*(beta*u_W[i] + g_W[i])
            b[i] = b[i] - eta*(beta*u_b[i] + g_b[i])
  
class rmsprop():
    
    def __init__(self):
        pass
    
    def update_rmsprop_params(self,eta,beta,epsilon):
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        W = self.params.W
        b = self.params.b
        u_W = self.u_params.W
        u_b = self.u_params.b
        N = len(W)
        for i in range(N):
            u_W[i] = beta*u_W[i] + (1-beta)*np.multiply(grad_w[i],grad_w[i])
            u_b[i] = beta*u_b[i] + (1-beta)*np.multiply(grad_b[i],grad_b[i])
            W[i] = W[i] - (eta*grad_w[i]/(np.sqrt(u_W[i]+epsilon)))
            b[i] = b[i] - (eta*grad_b[i]/(np.sqrt(u_b[i]+epsilon)))

class adam():
    
    def __init__(self):
        pass
    
    def update_adam_params(self,eta,beta1,beta2,epsilon):
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        W = self.params.W
        b = self.params.b
        u_W = self.u_params.W
        u_b = self.u_params.b
        m_W = self.m_params.W
        m_b = self.m_params.b
        N = len(W)
        for i in range(N):
            m_W[i] = beta1*m_W[i] + (1-beta1)*grad_w[i]
            u_W[i] = beta2*u_W[i] + (1-beta2)*np.square(grad_w[i])
            m_hat_W = m_W[i]/(1.0-np.power(beta1,i+1))
            u_hat_W = u_W[i]/(1.0-np.power(beta2,i+1))
            W[i] = W[i] - (eta/(np.sqrt(u_hat_W)+epsilon))*m_hat_W
            
            m_b[i] = beta1*m_b[i] + (1-beta1)*grad_b[i]
            u_b[i] = beta2*u_b[i] + (1-beta2)*np.square(grad_b[i])
            m_hat_b = m_b[i]/(1.0-np.power(beta1,i+1))
            u_hat_b = u_b[i]/(1.0-np.power(beta2,i+1))
            b[i] = b[i] - (eta/(np.sqrt(u_hat_b)+epsilon))*m_hat_b

class nadam():
    
    def __init__(self):
        pass
    
    def update_nadam_params(self,eta,beta,beta1,beta2,epsilon,X,Y):
        W = self.params.W
        b = self.params.b
        u_W = self.u_params.W
        u_b = self.u_params.b
        m_W = self.m_params.W
        m_b = self.m_params.b
        g_W = self.lookahead_params.W
        g_b = self.lookahead_params.b
        N = len(W)
        
        self.params.ForwardPropagation(X)
        self.params.BackPropagation(X,Y)
        
        grad_w = self.params.grad_w
        grad_b = self.params.grad_b
        
        for i in range(N):
            g_W[i] = grad_w[i]
            g_b[i] = grad_b[i]
            
        for i in range(N):
            m_W[i] = beta1*m_W[i] + g_W[i]
            m_b[i] = beta1*m_b[i] + g_b[i]
            
            u_W[i] = beta2*u_W[i] + (1-beta2)*np.square(grad_w[i])
            u_b[i] = beta2*u_b[i] + (1-beta2)*np.square(grad_b[i])
            
            m_hat_W = (beta1*m_W[i] + g_W[i])/(1-np.power(beta1,i+1))
            m_hat_b = (beta1*m_b[i] + g_b[i])/(1-np.power(beta1,i+1))
            u_hat_W = u_W[i]/(1-np.power(beta2,i+1))
            u_hat_b = u_b[i]/(1-np.power(beta2,i+1))
            
            W[i] = W[i] - (eta*m_hat_W/(np.sqrt(u_hat_W)+epsilon))
            b[i] = b[i] - (eta*m_hat_b/(np.sqrt(u_hat_b)+epsilon))


