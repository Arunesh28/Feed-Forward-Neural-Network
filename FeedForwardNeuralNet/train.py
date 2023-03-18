import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
import sklearn.model_selection
import wandb
import argparse
from activation import sigmoid,tanh,relu,der_sigmoid,der_tanh,der_relu,softmax,der_softmax
from NeuralNet import NeuralNet
dataset_dict = {'fashion_mnist' : fashion_mnist, 'mnist' : mnist}

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default="CS6910_Assignment_1")
    parser.add_argument('--wandb_entity', type=str, default='Arunesh J B')
    parser.add_argument('--dataset',type=str,default = 'fashion_mnist')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str,default='nadam')
    parser.add_argument('--learning_rate',type = float, default = 0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=0.00000001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--weight_init', type=str, default='xavier')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='ReLU')
    args = parser.parse_args()
    return args
    
def sweep_dict(args):
    return {
                "method" : "grid",
                "metric" :{
                    "name" : "validation_accuracy",
                    "goal" : "maximize"
                },
                "parameters" : {
                    "num_epochs" : {
                        "values" : [args.epochs]
                    },
                    "num_hidden_layers" : {
                        "values" : [args.num_layers]
                    },
                    "hidden_layer_size" : {
                        "values" : [args.hidden_size]
                    },
                    "weight_decay" : {
                        "values" : [args.weight_decay]
                    },
                    "learning_rate" : {
                        "values" : [args.learning_rate]
                    },
                    "optimizer" : {
                    "values" : [args.optimizer] 
                    },
                    "batch_size" : {
                        "values" : [args.batch_size]
                    },
                    "weight_initializer" : {
                        "values" : [args.weight_init]
                    },
                    "activation" : {
                        "values" : [args.activation]
                    },
                }
            }
    

    

def main(args):
    
    dataset = args.dataset
    (X, Y), (X_test, Y_test) = dataset_dict[dataset].load_data()
    
    #Data Cleaning
    num_features = np.shape(X)[1]*np.shape(X)[2]
    X = X/255.0
    X_test = X_test/255.0
    X = X.reshape(np.shape(X)[0], 784)
    X_test = X_test.reshape(np.shape(X_test)[0], 784)
    X_train, Xv, Y_train, Yv = sklearn.model_selection.train_test_split(X, Y, test_size=0.1, random_state=4, shuffle=True)
    X_train = X_train.T
    Xv = Xv.T
    X_test = X_test.T
    num_classes = len(np.unique(Y))
    
    sweep_config = sweep_dict(args)
    
    def train():
        wandb.init()
        config = wandb.config
        HIDDEN_LAYER_SIZE = args.hidden_size
        NUM_FEATURES = X_train.shape[0]
        WEIGHT_INITIALIZER = args.weight_init
        NUM_HIDDEN_LAYERS = args.num_layers
        HIDDEN_LAYER_DIMS = (args.hidden_size+np.zeros(NUM_HIDDEN_LAYERS)).astype(int)
        OPTIMIZER = args.optimizer
        LEARNING_RATE = args.learning_rate
        ACTIVATION = args.activation
        OUTPUT_LAYER_DIM = num_classes
        BATCH_SIZE = args.batch_size
        EPOCHS = args.epochs
        WEIGHT_DECAY = args.weight_decay
        
        run_name = "op_{}_ac_{}_wi_{}_lr_{}_bs_{}_l2_{}_nh_{}_sh_{}_ep_{}".format(OPTIMIZER,ACTIVATION,WEIGHT_INITIALIZER,LEARNING_RATE,BATCH_SIZE,WEIGHT_DECAY,NUM_HIDDEN_LAYERS,HIDDEN_LAYER_SIZE,EPOCHS)
        
        FFN = NeuralNet(num_features = NUM_FEATURES,
                        weight_initializer = WEIGHT_INITIALIZER,
                        num_hidden_layers = NUM_HIDDEN_LAYERS,
                        hidden_layer_dims = HIDDEN_LAYER_DIMS,
                        optimizer = OPTIMIZER,
                        learning_rate = LEARNING_RATE,
                        activation = ACTIVATION,
                        X_train = X_train,
                        Y_train = Y_train,
                        Xv = Xv,
                        Yv = Yv,
                        weight_decay=WEIGHT_DECAY,
                        output_layer_dim = OUTPUT_LAYER_DIM,
                        batch_size = BATCH_SIZE,
                        num_epochs = EPOCHS)
        FFN.fit_NeuralNet()
        Y_pred_test = FFN.predict_NeuralNet(X_test)
        print("Test Accuracy :",FFN.accuracy_score(Y_pred_test,Y_test))
        wandb.run.name = run_name
        wandb.run.save()
    
    sweep_id = wandb.sweep(sweep_config,project=args.wandb_project)
    wandb.agent(sweep_id=sweep_id,function=train,count = 1)
    
if __name__ == "__main__":
    args = parse()
    main(args)
    
 