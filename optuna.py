##Editor Qiuming Li
#Import Librarys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np
import os
import time

import optuna

#Import custom modules import net and multiple unet， import unetdataset and multiple unet dataset
from neural_networks import UNET, MultiInput_UNET
from data_set import UNET_Dataset, MI_Dataset

## time estimation to estimate the time of each trial
def time_estimation(start_time, factor = 1, print_time = True):
    
    ##Overview: A helper Function which will print the time, how long the task will take
    ##start_time: time, when the original process started
    ##factor: if you run the original process multiple times, this can be mutliplied
    ##print_time: will print the time in terminal
    ##delta: time difference
    ##unit: unit of time difference (seconds, minutes, hours)
    
    end_time = time.time()
    delta = end_time - start_time
    delta *= factor
    unit = "seconds"
    
    #Transforms the Data into Minutes if you have more than 60. Seconds
    if delta >= 60.0:
        delta /= 60
        unit = "minutes"
    
    #Transforms the Data into Hours if you have more than 60. Minutes
    if delta >= 60.0:
        delta /= 60
        unit = "hours"
        
    if print_time:
        print("##TIME ESTIMATION##\nThe Programm needs {} {} to be executed.\nThis is only displayed once.".format(round(delta,2), unit))
        
    return delta, unit

def global_data_init(data, train_idx, val_idx):
    
    ##Overview: A helper Function which will init global variables. Global variables will be used inside the Optimization process
    ##data: the complete data as dataset
    ##train_idx: indexes of the data which will be used during training
    ##val_idx: indexes of the data which will be used during validation
    
    global data_set
    global train_sampler
    global val_sampler
    
    data_set = data
    train_sampler = train_idx
    val_sampler = val_idx
    

def train_HPO(model, dataloader, criterion, optimizer, device):

    ##Overview: A Train Function which will used for training our model. 
    ##model: the fully initialized pytorch nerual network
    ##dataloader: pytroch Dataloader with training-data
    ##Criterion: a criterion how the Loss will be calculated (MSELoss)
    ##Optimizer: An Optimizer which will be used for backpropagation (for example Adam)
    ##device: pytorch device where you want to load the data ("cuda" or "cpu") here we run on the cuda , means gpu
    ##epoch_loss: the loss of the epoch, calculated with the criterion
    
    epoch_loss = 0.0
    
    #Sets the Model on Training State
    model.train()
    
    #Start Trianing
    #for X, y_true in dataloader:
    for X, y_bad, y_true in dataloader:
        model.zero_grad()

        X = torch.autograd.Variable(X.float())        
        y_true = torch.autograd.Variable(y_true)
        y_bad = torch.autograd.Variable(y_bad)
        
        #Pack the Values onto the GPU or CPU
        X, y_true, y_bad = X.to(device), y_true.to(device), y_bad.to(device)

        #Predict Value
        y_pred = model(X.float(), y_bad.float())
        y_pred = y_pred.to(device)
        
        #Calculate loss
        batch_loss=criterion(y_pred.double(),y_true.double())
        epoch_loss += batch_loss.item() /len(dataloader)

        #Backpropagate and Optimize
        batch_loss.backward()
        optimizer.step()
        
        #Free gpu memory
        del X, y_true, y_pred, y_bad
        torch.cuda.empty_cache()
    return epoch_loss

def val_HPO(model, dataloader, criterion, device):

    ##Overview: A Validation Function which will used for Validation our model. 
    ##model: the fully initialized pytorch nerual network
    ##dataloader: pytroch Dataloader with training-data
    ##Criterion: a criterion how the Loss will be calculated (MSELoss)
    ##device: pytorch device where you want to load the data ("cuda" or "cpu")
    ##val_loss: the loss of the validation, calculated with the criterion


    #VALIDATION
    model.eval()
    eval_loss = 0.0
    with torch.no_grad(): #In this Context -> No Adjustment of gradients
        #for X, y_true in dataloader:
        for X, y_bad, y_true in dataloader:
            
            X = torch.autograd.Variable(X.float())
            y_true = torch.autograd.Variable(y_true)
            y_bad = torch.autograd.Variable(y_bad)
            
            #Pack the Values onto the GPU or CPU
            X, y_true, y_bad = X.to(device), y_true.to(device), y_bad.to(device)
            
            #Predict Value
            y_pred = model(X.float(), y_bad.float())
            y_pred = y_pred.to(device)
            
            eval_loss += criterion(y_pred.double(), y_true.double()).item() / len(dataloader)
            
            #Free gpu memory
            del X, y_true, y_pred, y_bad
            torch.cuda.empty_cache()
    return eval_loss


def HPO_UNET_Optuna(trial): 
    
    ##Overview: The Function which Optuna tries to optimize for the Unet. Only Optuna calls this Function
    ##trial: this is from optuna. It choose a parameter for the params we suggest with trial.suggest...

    #Check for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Init all values which will be optimized by Optuna
    batch_size = trial.suggest_categorical('batch_size', [4,8,16])## the value of batch_size is 4,8,16
    kernel = trial.suggest_categorical('kernel', [3])## the value of kernel_size is 3
    activation_name = trial.suggest_categorical('activation', ["relu", "tanh", "sigmoid"])##the value of activation
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)## the value of dropout_rate, is a intervel
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)## the value of learning_rate. is a intervel
    
    #Short Printing which Values are selted during this trial
    print("###############")
    print("""Trial startet:
    Device: {}
    - Batch-Size: {}
    - Kernel:     {}
    - Activation: {}
    - dropout:    {}
    - learning:   {}""".format(
        device, 
        batch_size,
        kernel, 
        activation_name, 
        dropout_rate, 
        learning_rate))
    print("###############")
    
    #Dataloader
    train_loader = DataLoader(data_set, batch_size = batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(data_set, batch_size = batch_size, sampler=val_sampler, drop_last=True)
    
    #Model
    model = UNET(in_features  = 4, ## input is 4 dimensions
                    out_features = 1, 
                    batch_size   = batch_size,
                    kernel       = kernel,
                    activation   = getattr(torch, activation_name),
                    dropout_rate = dropout_rate)
    
    #Optimizer and Criterions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    #Set to GPU
    model = model.to(device)
    criterion = criterion.to(device)
    
    #Start Evaluating
    val_loss = 0.0
    ret_val = 0.0
    epochs = 25
    
    #Time Estimation ,use time estimation to guess the time to run the programm
    start_time = time.time()
    
    for epoch in range(epochs):
        if epoch == 1:
            time_estimation(start_time, factor = epochs - 1)
            
        print("Starting Epoch [{}/{}]".format(epoch+1, epochs))
        
        #Train 
        train_HPO(model, train_loader, criterion, optimizer, device) 
        
        #Val 
        val_loss = val_HPO(model, val_loader, criterion, device)
        ret_val += val_loss
        
        #Check for prune
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            print("###Trial will be pruned. Epochs [{}/{}]###".format(epoch+1, epochs))
            raise optuna.TrialPruned()
            
        print("Validation loss: {} | Combined Return Value: {}".format(round(val_loss,6), round(ret_val, 6)))
        
       
    #free gpu
    del model
    del criterion
    torch.cuda.empty_cache()
            
    return ret_val

def HPO_MI_Optuna(trial):

    ##Overview: The Function which Optuna tries to optimize for the MultiInput-Unet. Only Optuna calls this Function
    ##trial: this is from optuna. It choose a parameter for the params we suggest with trial.suggest...
    
    #Check for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Init all values which will be optimized by Optuna
    batch_size = trial.suggest_categorical('batch_size', [4,8,16])
    kernel = trial.suggest_categorical('kernel', [3])
    activation_name = trial.suggest_categorical('activation', ["relu", "tanh", "sigmoid"])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    #Short Printing which Values are selted during this trial
    print("###############")
    print("""Trial startet:
    Device: {}
    - Batch-Size: {}
    - Kernel:     {}
    - Activation: {}
    - dropout:    {}
    - learning:   {}""".format(
        device, 
        batch_size,
        #filters, 
        kernel, 
        activation_name, 
        dropout_rate, 
        learning_rate))
    print("###############")
    
    #Dataloader
    train_loader = DataLoader(data_set, batch_size = batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(data_set, batch_size = batch_size, sampler=val_sampler, drop_last=True)
    
    #Model
    model = MultiInput_UNET(in_features  = 3, ##ibput is 3 dimensions
                    out_features = 1, 
                    batch_size   = batch_size,
                    #filters      = filters,
                    kernel       = kernel,
                    activation   = getattr(torch, activation_name),
                    dropout_rate = dropout_rate)
    
    #Optimizer and Criterions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    #Set to GPU
    model = model.to(device)
    criterion = criterion.to(device)
    
    #Start Evaluating
    val_loss = 0.0
    ret_val = 0.0
    epochs = 25
    
    #Time Estimation
    start_time = time.time()
    
    for epoch in range(epochs):
        if epoch == 1:
            time_estimation(start_time, factor = epochs - 1)
            
        print("Starting Epoch [{}/{}]".format(epoch+1, epochs))
        
        #Train
        train_HPO(model, train_loader, criterion, optimizer, device)
        
        #Val
        val_loss = val_HPO(model, val_loader, criterion, device)
        ret_val += val_loss
        
        #Check for prune
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            print("###Trial will be pruned. Epochs [{}/{}]###".format(epoch+1, epochs))
            raise optuna.TrialPruned()
            
        print("Validation loss: {} | Combined Return Value: {}".format(round(val_loss,6), round(ret_val, 6)))
        
       
    #free gpu
    del model
    del criterion
    torch.cuda.empty_cache()
            
    return ret_val

def start_HPO_Optuna(HPO_Func, num_trials, study_name, storage_path, del_storage_if_exist=True, load_if_exists=True):
    
    ##Overview: A Function which initializes the Hyperparameter-Optimization (HPO) and starts the optimization. 
    ##HPO_Func: which Function will be optimized
    ##num_trials: How many trials can the Optiimizer maker
    ##study_name: name of the study which will be created
    ##storage_path: path of the study where it will be safed
    ##del_storage_if_exist: deltes the storage if there is already a study
    ##load_if_exists: checks if the under the storage_path is already a study. 
                      
    
    #Checks if the global values are set
    if 'data_set' not in globals():
        print("No Global data initialized.\nRun global_data_init(data, train_idx, val_idx) with appropriate values")
        return
    
    else:
        print(data_set)
    
    #Giving Feedback, which Combination will be used.
    if del_storage_if_exist and os.path.exists(storage_path):
        print("Storage {} already exists - will be deleted".format(storage_path))
        os.remove(storage_path)
    elif os.path.exists(storage_path) and not load_if_exists:
        print("Storage {} already exists and will not be overwritten.".format(storage_path))
        return
    
    #Create the Study
    study = optuna.create_study(
        study_name=study_name,
        storage='sqlite:///' + storage_path,
        direction = 'minimize',
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=10,
                                           interval_steps=2))
    #Start Optimization
    study.optimize(HPO_Func, n_trials=num_trials)
    
    print("Best config: \n", study.best_trial)


def HPO_Functions():
    
    ##Overview: A Function which returns all available Functions inside an dictionary

    ##A Dictionary with all Optimization Functions

    return {
        "unet_op" : HPO_UNET_Optuna,
        "mi_op": HPO_MI_Optuna,
    }
    
def main():
    """
    Overview: A main Function so the optimization dont starts by accident
    """
    print("Starting HPO")
    print("------------")
    
    print("Extrakting Data")
    #Get Data
    root = r"C:/Users/49162/Desktop/python/images/"
    camera_dir = r"camera/"
    sensor_dir = r"depth_bad/"
    masks_dir = r"depth_real/"

    #For Unet
    #data_set = UNET_Dataset(root, camera_dir, sensor_dir, masks_dir) #NO one-hot encoding
    #For Multiinput
    data_set = MI_Dataset(root, camera_dir, sensor_dir, masks_dir) #NO one-hot encoding

    #Sample Data
    #Variables
    train_size = 0.8
    test_size = 0.1
    val_size = 0.1

    #shuffle indexes
    sample_idx = [i for i in range(len(data_set))]
    np.random.shuffle(sample_idx)

    len_train = int(len(data_set) * train_size)
    len_val = int(len(data_set) * val_size)

    train_idx = sample_idx[:len_train]
    val_idx = sample_idx[len_train:len_train+len_val]
    test_idx = sample_idx [len_train+len_val:]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    #Short Overview at the beginning of the optimization
    print("""Got Data:
     - Number of Data:{}
     - Number of Traing-Values: {}
     - Number of Validation-Values: {}
     - Number of Test-Values: {}""".format(len(data_set), len(train_idx), len(val_idx), len(test_idx)))

    #Init and start HPO 
    hpo_funcs = HPO_Functions()
    global_data_init(data_set, train_sampler, val_sampler)
    
    start_HPO_Optuna(
        hpo_funcs["mi_op"], 
        num_trials = 150, 
        study_name = 'mi_study', ##to create a new study name
        storage_path='HPO/optuna_db/mi_hpo.db',## to store all trials
        del_storage_if_exist=False,
        load_if_exists=True
        )

if __name__ == "__main__":
    main()







