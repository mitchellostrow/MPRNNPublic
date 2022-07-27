import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.ndimage import gaussian_filter

from mprnn.testing import test_specific_opponents
from mprnn.utils import convert_dist_to_params, get_env_from_json, get_net, FILEPATH

def define_decoding_test_opponents():
    '''
    reads a JSON file in MPRNNPublic/data and get the list of opponents we're testing again
    returns output list of tuples (str,dict) with opponent name and parameter dict
    Returns:
        test_opponents (list(tuple(str,dict))): details each opponent name and set of parameters
    '''
    path = Path(FILEPATH,'data','linear_decoding_opponents.json')
    with open(path) as f:
        opponents = json.load(f)

    test_opponents = []
    for i,opp in enumerate(opponents['opponents']):
        test_opponents.append( (opp,opponents['parameters'][i]) )
    return test_opponents      

def train_test_split(prewashoutblocks,postwashoutblocks, train_split=0.7):
    '''
    Given the TrialTestObjects in data, extract the state and class for linear classification of state to class.
    Arguments:
        prewashoutblocks list(TrialTestObject): trial data from before the washout
        postwashoutblocks list(TrialTestObject): trial data from after the washout
        train_split (float): ranging from 0 to 1, represents how much of the prewashoutblock to use as training data
    Returns:
        lists of states for train_within, test_within, test_between (X)
        lists of opponent labels for the above (y)
    '''
    L = len(prewashoutblocks[0].states)
    train = int(L*train_split)

    def get_x_y_(blockdata,startindex,stopindex):
        y,X = [],[]
        for block in blockdata:
            labels = np.array(block.opponent_indices[startindex:stopindex])
            labels[-1] = labels[-2] #fix a bug where the last label switches (referencing the new opponent)
            y.extend(labels)

            states = block.states[startindex:stopindex]
            X.extend(states[:,:states.shape[-1]//2]) #only use the hidden state, not the cell state
        return X,y

    X_train,y_train = get_x_y_(prewashoutblocks,0,train)
    X_test_within,y_test_within = get_x_y_(prewashoutblocks,train,L)
    X_test_between,y_test_between = get_x_y_(postwashoutblocks,0,L)

    return y_train,y_test_within,y_test_between,X_train,X_test_within,X_test_between

def get_decoding_train_test(env, model, opponents,nblocks=1,nwashout=30,train_split = 0.7):
    '''
    tests the agent with the test_specific_opponents function and separates
    into train+test data for both within block and between block
    Arguments: 
        same as test_specific_opponents
        train_split (float between 0 and 1): split of the train_test split
        nblocks (int): how many blocks to use as data.
        nwashout (int): number of blocks to put in betwen first and last set of opponents
    Returns:
    '''
    train_test_data = OrderedDict({"y_train":[],"y_test_within":[],"y_test_between":[],
                        "X_train":[],"X_test_within":[],"X_test_between":[]})
    for _ in range(nblocks):
        prewashout_blocks, postwashout_blocks = test_specific_opponents(env,model,opponents,nwashout)
        single_block_data = train_test_split(prewashout_blocks,postwashout_blocks,train_split)
        for i,all_data in enumerate(train_test_data.values()):
            all_data.extend(single_block_data[i])
    return train_test_data

def train_classifier(X,y,classifier_type="logistic",penalty="l2",C=1.0):
    '''
    trains a classifier on X and y data
    Returns:
        clf (SVC or LogisticRegression): a trained classifier
    '''
    if classifier_type == "SVC":
        clf = SVC() 
    else:
        clf = LogisticRegression(penalty=penalty,solver='liblinear',fit_intercept=False,C=C)
    #right now this will be a list of 2 (within and between)
    clf.fit(X,y)
    return clf

def get_data_classify(args):
    '''
    Given specification for a model and trial arguments in args, get the right environment, model, and test agent on env
    Then train a classifier and return it
    Arguments:
        args (args.argParse object):
            trun (int): code for the model run 
            trainiters (str): length of training for the particular model
            mtype (str): type of model (SSP or default)
            nwashout (int): how many blocks to use as the washout between blocks
            nblocks (int): how many blocks to test the agent
            clf ("logistic","SVC"): logistic regression or support vector classifier
            penalty ("l2", "l1", "none"): regularization term for model
            train_split (float): percent of prewashout_trials to use as training data
    Returns:
        train_test_data (dict): all training and testing data stored in a dict
        clf (SVC or LogisticRegression): a trained classifier
    '''
    model,_ = get_net(args.trun,args.mtype,args.trainiters)  
    with open(Path(FILEPATH,"data","models",f"run{args.trun}",Path("train_params.json") ) ) as f:
        script_kwargs = json.load(f)
    train_params = convert_dist_to_params(script_kwargs)
    env = get_env_from_json(train_params['env'],useparams=True) 
    test_opponents = define_decoding_test_opponents()

    train_test_data = get_decoding_train_test(env, model, test_opponents,args.nblocks,args.nwashout,args.train_split)
    numopps = len(test_opponents)
    train_test_data['num_opps'] = numopps

    clf = train_classifier(train_test_data['X_train'],train_test_data['y_train'],args.clf,args.penalty)
    return clf, train_test_data

def block_test(X_train,X_test,y_train,y_test,clf):
    '''
    Given train test data and a classifier, tests the classifier and returns its predictions
    Returns:
        acc_train,acc_test (np.arrays): lists of whether or not the classifier was accurate on individual trials over time
    '''
    def pred(X,y):
        pred = clf.predict(X)
        acc = np.array(pred == y, dtype=np.int)   
        #right now this is data from block 1-3 concatenated
        return acc
    acc_train = pred(X_train,y_train)
    acc_test = pred(X_test,y_test)
    return acc_train, acc_test

def get_clf_accuracy_over_time(train_test_data,clf,nblocks):
    '''
    Given saved train/test data and a trained classifier, as well as the number of test blocks,
    Calculates the mean classifier acuracy on all datasets over block time
    Arguments:
        train_test_data (dict): data of the trials, split into train and test and within/between
        clf (SVC,logisticregression): trained classifier
        nblocks (int): from arguments, number of blocks in dataset
    Returns:
        mu_train,mu_test_within,mu_test_between (np.array): mean classifier accuracy at a particular time in the block
        stderr_train, ... (np.array): standard error of the mean at a particular time in the block
    '''
    numopps = train_test_data['num_opps']

    acc_train, acc_test_within = block_test(train_test_data['X_train'],train_test_data['X_test_within'],
                                            train_test_data['y_train'],train_test_data['y_test_within'],
                                            clf)
    _, acc_test_between = block_test(train_test_data['X_train'],train_test_data['X_test_between'],
                                            train_test_data['y_train'],train_test_data['y_test_between'],
                                            clf)
    def get_mean_stderr(data):
        data = data.reshape(numopps*nblocks,data.shape[0]//(numopps*nblocks)) #separate by block
        #data.shape[0]//numopps*nblocks should be equal to 150
        stderr = data.std(axis=0)/np.sqrt(data.shape[0]) #average over the first column
        mu = data.mean(axis=0)
        return mu,stderr
    mu_train,stderr_train = get_mean_stderr(acc_train)
    mu_test_within,stderr_test_within = get_mean_stderr(acc_test_within)
    mu_test_between,stderr_test_between = get_mean_stderr(acc_test_between)

    return mu_train,mu_test_within,mu_test_between,stderr_train,stderr_test_within,stderr_test_between

def plot_errorbar(x,y,err,ax=None,color="tab:blue"):
    '''
    Makes an error bar in the plot, above and below the x,y chart
    '''
    upper = y + 0.5 *err
    lower = y - 0.5*err
    if ax is None:
        p = plt
    else:
        p = ax
    p.plot(x,upper,color=color,alpha=0.1)
    p.plot(x,lower,color=color,alpha = 0.1)
    p.fill_between(x,lower,upper,alpha=0.2,color=color)

def plot_acc(mu_train,mu_test_within,mu_test_between,stderr_train,stderr_test_within,stderr_test_between):
    '''
    Given train and test data, plots!
    Arguments:
        output of get_clf_accuracy_over_time
    Returns:
        None
    '''
    mu_train = gaussian_filter(mu_train,sigma=2)
    mu_test_within = gaussian_filter(mu_test_within,sigma=2)
    mu_test_between = gaussian_filter(mu_test_between,sigma=2)
    _, ax = plt.subplots(1,1,figsize=(3.5,2.5))
    len_train = len(mu_train)
    len_block = len(mu_test_between)

    ax.plot(np.arange(0,len_train),mu_train,label="Within-Block Train",c="#DC640F")
    ax.plot(np.arange(len_train,len_block),mu_test_within+1e-2,label="Within-Block Test",c="#D139AE")

    plot_errorbar(np.arange(0,len_train),mu_train,stderr_train,ax,color="#DC640F")
    plot_errorbar(np.arange(len_train,len_block),mu_test_within+1e-2,stderr_test_within,ax,color="#DC640F")

    ax.plot(np.arange(0,len_block),mu_test_between,label="Between-Block Test",c="#467100")
    plot_errorbar(np.arange(0,len_block),mu_test_between,stderr_test_between,ax,color="#467100")

    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Classifier Accuracy")
    ax.set_ylim([0.2,1.05])
    plt.legend()
    plt.tight_layout()  
    plt.savefig(Path("results","Decoding.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and Test the Agent's Performance, then \
                                                trains, tests, and plots the results of a linear classifier")

     
    parser.add_argument("--trun", type=int, required=False, default=86, help="model run # of the \
                                                                        trained multi-task agent")
    parser.add_argument("--mtype", required=False, default = "SSP", help="type of model (SSP or notSSP)")
    
    parser.add_argument("--trainiters", required=False, default="8000k", help="length of model training to be used (number+k)")
    parser.add_argument("--nblocks",type=int, required=False, default="50", help="number of trial blocks to test each opponent on")
    
    
    parser.add_argument("--nwashout",type=int,required=False,default = 30,help="length of washout blocks")
    parser.add_argument("--clf",required=False,default= "logistic", help = "'logistic','SVC'): logistic regression or support vector classifier")
    parser.add_argument("--penalty",default="l2",required = False, help = "regularization term for linear classifier")
    parser.add_argument("--train_split",default=0.7,type=float,help = "percent of prewashout_trials to use as training data")

    args = parser.parse_args()   

    clf, train_test_data = get_data_classify(args)
    plot_acc(*get_clf_accuracy_over_time(train_test_data,clf,args.nblocks))