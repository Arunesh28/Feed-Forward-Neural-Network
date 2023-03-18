import numpy as np
import wandb
from Parameters import Parameters
from activation import sigmoid,tanh,relu,der_sigmoid,der_tanh,der_relu,softmax,der_softmax
from optimizers import sgd,momentum,rmsprop,adam,nesterov,nadam

class NeuralNet:
    '''
        weight_initializers : dictionary with random , xavier
        weight_initializer : function
        activation_funtions : dictionary with sigmoid, tanh, relu
        der_activation_functions : dictionary with derivatives of the above
        optimizer_funtions : dictionary with sgd, momentum, nesterov, rmsprop, adam, nadam}
        activation : string
        opitmizer : string
        learning_rate : int
        batch_size : int
        num_epochs : int
        num_features : dimension of X
        num_hidden_layers : int, number of hidden layers
        output_layer_dim : int
        hidden_layer_dims : np.array with num_neurons in all hidden layer 
        weight_Decay : L2 regularisation
        X_train : Training Data (n,d)
        Y_train : Training Data (n,)
        Xv : Validation Data (n,d)
        Yv : Validation Data (n,)
        hidden_layers : np.array of objects to class hidden_layer dimensions = num_hidden_layers
        output_layer : object to hidden_layer class
        params : It is a object of class parameter which has the weights and biases of the neural network
        optimizer_object : object to the optimizer class, initialised in the initialize_neuralnet funciton
        beta : momentum,neterov,rmsprop,nadam
        beta1 : adam,nadam
        beta2 : adam,nadam
        epsilon : rmsprop,adam,nadam
    '''
    def __init__(self,
                 num_features,
                 weight_initializer,
                 num_hidden_layers,
                 hidden_layer_dims,
                 optimizer,
                 learning_rate,
                 activation,
                 X_train,
                 Y_train,
                 Xv,
                 Yv,
                 weight_decay,
                 output_layer_dim,
                 batch_size,
                 num_epochs,
                 loss = 'cross_entropy',
                 output_activation = softmax,
                 der_output_activation = der_softmax,
                 beta=0.9,
                 epsilon=1e-8,
                 beta1=0.9,
                 beta2=0.999):
        self.weight_initializers = {"random": self.random_initialization, "xavier": self.xavier_intialization}
        self.weight_initializer = self.weight_initializers[weight_initializer]
        self.activation_functions = {"sigmoid": sigmoid, "tanh": tanh, "ReLU": relu}
        self.der_activation_functions = {"sigmoid": der_sigmoid, "tanh": der_tanh, "ReLU": der_relu}
        self.loss_functions = {'cross_entropy':self.Cross_Entropy_Loss,'mse' : self.Square_Error_Loss}
        self.loss = self.loss_functions[loss]
        '''
            Add your optimizer function and class in the below dictionaries
        '''
        self.optimizer_functions = {"sgd": self.sgd, "momentum": self.momentum,"nesterov": self.nesterov, "rmsprop": self.rmsprop, "adam": self.adam, "nadam": self.nadam}
        self.optimizer_classes = {"sgd": sgd, "momentum": momentum,"nesterov": nesterov, "rmsprop": rmsprop, "adam": adam, "nadam": nadam}
        self.activation = self.activation_functions[activation]
        self.optimizer = self.optimizer_functions[optimizer]
        self.optimizer_class = self.optimizer_classes[optimizer]
        self.der_activation = self.der_activation_functions[activation]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.output_layer_dim = output_layer_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.num_features = num_features
        self.output_activation = output_activation
        self.der_output_activation = der_output_activation
        self.weight_decay = weight_decay
        self.X_train = X_train
        self.Y_train = Y_train
        self.old_Y_train = Y_train
        self.Xv = Xv
        self.Yv = Yv
        self.old_Yv = Yv
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.num_classes = self.output_layer_dim
        # return self
        
    def Square_Error_Loss(self,Y_pred,Y_actual):
        return np.mean((Y_pred-Y_actual)**2)
    
    def Cross_Entropy_Loss(self,Y_pred,Y_actual):
        return (-1.0*np.sum(np.multiply(Y_actual+1,np.log(Y_pred+1))))/float(Y_pred.shape[0])
    
    def random_initialization(self, in_layer, out_layer):
        return np.random.randn(in_layer, out_layer)

    def xavier_intialization(self, in_layer, out_layer):
        return np.random.randn(in_layer, out_layer)* np.sqrt(2 / (in_layer + out_layer))
    
    '''
        The below optimizer functions are used to call the update_params methods in of their respective params
    '''
    
    def sgd(self,X,Y):
        sgd_obj = self.optimizer_object
        parameters = self.params
        parameters.ForwardPropagation(X)
        parameters.BackPropagation(X,Y)
        sgd_obj.update_sgd_params(self.learning_rate)
        # for i in range(len(self.params.W)):
        #     print(self.params.W[i])
    
    def momentum(self,X,Y):
        momentum_obj = self.optimizer_object
        parameters = self.params
        parameters.ForwardPropagation(X)
        parameters.BackPropagation(X,Y)
        momentum_obj.updade_momentum_params(self.learning_rate,self.beta)

    def nesterov(self,X,Y):
        nesterov_obj = self.optimizer_object
        nesterov_obj.update_nesterov_params(self.learning_rate,self.beta,X,Y)
    
    def rmsprop(self,X,Y):
        rmsprop_obj = self.optimizer_object
        parameters = self.params
        parameters.ForwardPropagation(X)
        parameters.BackPropagation(X,Y)
        rmsprop_obj.update_rmsprop_params(self.learning_rate,self.beta,self.epsilon)
    
    def adam(self,X,Y):
        adam_obj = self.optimizer_object
        parameters = self.params
        parameters.ForwardPropagation(X)
        parameters.BackPropagation(X,Y)
        adam_obj.update_adam_params(self.learning_rate,self.beta1,self.beta2,self.epsilon)
    
    def nadam(self,X,Y):
        nadam_obj = self.optimizer_object
        nadam_obj.update_nadam_params(self.learning_rate,self.beta,self.beta1,self.beta2,self.epsilon,X,Y)
    
    def optimzizer_name(self,X,Y):
        '''
            May implement forward and backpropagation here 
            Add code to create an object to your optimizer class
            Then use this object to call your optimizer_update method 
        '''
        pass
    
    def sanitize_Y(self):
        
        temp = np.zeros((self.num_classes,self.Y_train.shape[0]))
        for i in range(self.Y_train.shape[0]) :
            temp[int(self.Y_train[i])][i] = 1
        self.Y_train = temp
        
        temp = np.zeros((self.num_classes,self.Yv.shape[0]))
        for i in range(self.Yv.shape[0]) :
            temp[int(self.Yv[i])][i] = 1
        self.Yv = temp
        
    def initialize_NeuralNet(self):
        self.sanitize_Y()
        self.optimizer_object = self.optimizer_class()
        self.params = self.initialize_parameters()
        self.optimizer_object.params = self.params
        self.optimizer_object.u_params = self.zero_initializer()
        self.optimizer_object.m_params = self.zero_initializer()
        self.optimizer_object.lookahead_params = self.zero_initializer()
        '''
            can add required optimizer params and 
            initialize them here 
        '''
        # pass
    
    
    def add_hidden(self,num_neurons,pos):
        '''
        This function can be used to insert a layer at any position
        (0 indexing)
        '''
        self.hidden_layer_dims.insert(pos,num_neurons)
        self.num_hidden_layers += 1
        
    def zero_initializer(self):
        params = Parameters()
        W = []
        b = []
        in_layer, out_layer = self.num_features,self.hidden_layer_dims[1]
        for i in range(self.num_hidden_layers):
            if  i == 0:
                in_layer = self.num_features
            else:
                in_layer = self.hidden_layer_dims[i-1]
                
            if i == self.num_hidden_layers-1:
                out_layer = self.output_layer_dim
            else:
                out_layer = self.hidden_layer_dims[i+1]
            W.append(np.zeros(shape = (self.hidden_layer_dims[i],in_layer)))
            b.append(np.zeros(shape = (self.hidden_layer_dims[i],1)))
        W.append(np.zeros(shape = (self.num_classes,self.hidden_layer_dims[self.num_hidden_layers-1])))
        b.append(np.zeros(shape=(self.num_classes,1)))
        params.W = W
        params.b = b
        params.activation = self.activation
        params.output_activation = self.output_activation
        params.der_activation = self.der_activation
        params.weight_decay = self.weight_decay
        return params
    
    def initialize_parameters(self):
        params = Parameters()
        W = []
        b = []
        in_layer, out_layer = self.num_features,self.hidden_layer_dims[1]
        for i in range(self.num_hidden_layers):
            if  i == 0:
                in_layer = self.num_features
            else:
                in_layer = self.hidden_layer_dims[i-1]
                
            if i == self.num_hidden_layers-1:
                out_layer = self.output_layer_dim
            else:
                out_layer = self.hidden_layer_dims[i+1]
            W.append(self.weight_initializer(self.hidden_layer_dims[i],in_layer))
            b.append(np.zeros(shape=(self.hidden_layer_dims[i],1)))
        W.append(self.weight_initializer(self.num_classes,self.hidden_layer_dims[self.num_hidden_layers-1]))
        b.append(np.zeros(shape=(self.num_classes,1)))
        params.W = W
        params.b = b
        params.activation = self.activation
        params.output_activation = self.output_activation
        params.der_activation = self.der_activation
        params.weight_decay = self.weight_decay
        return params
    
    def fit_NeuralNet(self):
        self.initialize_NeuralNet()
        for curr_epoch in range(self.num_epochs):
            print("Epoch Number : ",curr_epoch+1)  
            for i in range(0,self.X_train.shape[1],self.batch_size):
                curr_batch = min(self.X_train.shape[1]-i,self.batch_size)
                self.optimizer(self.X_train[:,i:i+curr_batch],self.Y_train[:,i:i+curr_batch])
            train_acc,validation_acc,training_loss,validation_loss = self.accuracy_NeuralNet(self.old_Y_train,self.old_Yv)
            wandb.log({ "training_accuracy" : train_acc,
                        "validation_accuracy" : validation_acc,
                        "training_loss" : training_loss,
                        "validation_loss" : validation_loss,
                        "epoch" : curr_epoch+1})
            print(train_acc,validation_acc,training_loss,validation_loss)
        # pass
    
    def predict_NeuralNet(self,X):
        self.params.ForwardPropagation(X)
        Y_pred = np.argmax(self.params.H[len(self.params.H)-1],axis=0)
        return Y_pred
    
    def accuracy_NeuralNet(self,Y_train,Yv):
        Y_train_pred = self.predict_NeuralNet(self.X_train)
        Yv_pred = self.predict_NeuralNet(self.Xv)
        training_Loss = self.loss(Y_train_pred,self.old_Y_train)
        validation_Loss = self.loss(Yv_pred,self.old_Yv)
        return self.accuracy_score(Y_train_pred,Y_train),self.accuracy_score(Yv_pred,Yv),training_Loss,validation_Loss
    
    def accuracy_score(self, Y_pred, Y_train):
        return np.sum(Y_pred == Y_train)/Y_train.shape[0]
