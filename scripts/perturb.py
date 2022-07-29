import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from lineardecoding import get_data_classify,define_decoding_test_opponents,predict_opponent_class,plot_errorbar
from mprnn.utils import FILEPATH, default_argparser, get_net,setup_run,convert_dist_to_params,get_env_from_json,set_plotting_params
from mprnn.testing import test_specific_opponents
rng = np.random.default_rng()
set_plotting_params()

def get_axes(args):
    '''
    Given an agent code, runs a linear decoding test to get the axes and the classifier
    '''
    clf, train_test_data = get_data_classify(args)
    Xs = np.array(train_test_data['X_test_between'])
    ys = np.array(train_test_data['y_test_between'])
    decoding_coefs = clf.coef_ #3 x n_states
    for i in range(decoding_coefs.shape[0]):
        decoding_coefs[i] /= np.linalg.norm(decoding_coefs[i]) # make a basis
    return clf, decoding_coefs, Xs,ys

def get_perturb_from_json():
    '''
    Loads the perturb_dict parameters from a json file
    '''
    path = Path(FILEPATH,'data','perturb_dict.json')
    with open(path) as f:
        perturb_dict = json.load(f)
    return perturb_dict

def apply_perturbation(setup_run_dict,perturb_dict):
    '''
    Arguments:
        setup_run_dict (dict): specifies details of the model and the environment
        perturb_dict (dict): specifies how to perturb
            'time': int-number of step to apply perturbation
            'opp':int-which opponent to apply to 
            'decoding_coefs':np.array the list of opponent linear classifier coding bases (the coefficients of the classifier)
            'new_coefs_to_perturb_to': list (of length 3)- coefficients of the linear combination of params in the above space
            'reset': bool: True if we want to reset activity on this axis to the provided value, false if we just want to add it
    Returns:
        states,classes, rewards (np.array): perturbed RNN states, opponent classes, agent rewawrds
    '''
    _,_,nrec = setup_run(setup_run_dict['model'])

    #params correspond to coefficients for opponent 0, 1, 2,
    #so if we want to reset to only have a particular value for opponent 1 (as specified positionally)
    #then the coefficients are 0,1,0 
    perturb_dict['new_coefs_to_perturb_to'] = np.array(perturb_dict['new_coefs_to_perturb_to'])
    perturb_vector = perturb_dict['new_coefs_to_perturb_to'].T @ perturb_dict['decoding_coefs'] 
    perturb_dict['vector'] = perturb_vector 

    _,postwashout_trials = test_specific_opponents(setup_run_dict['env'],setup_run_dict['model'],setup_run_dict['test_opponents'],1,perturb_dict)
    states = np.array([block.states[:,:nrec//2] for block in postwashout_trials])
    classes = np.array([block.opponent_indices for block in postwashout_trials])
    rewards = np.array([block.rewards for block in postwashout_trials])
 
    return states,classes,rewards

def run_multiple_perturb_trials(args,setup_run_dict,perturb_dict):
    '''
    Given the details for how to perturb and how to run the model,
    Runs the perturbation trial multiple times, specified by args.nperturb, and saves all data
    Arguments:
        setup_run_dict (dict): details of the run (environment, model, etc)
        args (ArgumentParser): input arguments from command line
        perturb_dict (dict): read from perturb_dict.json, specifies how to perturb
    Returns:
        perturb_trial_data_dict (dict): contains data from perturbation trials, including hidden states, new linear
                                        decoding coefs, classes, rewards, classifier, and the original perturb_dict
    '''
    clf,decoding_coefs,old_Xs,_ = get_axes(args)
    perturb_dict['clf'] = clf
    perturb_dict['decoding_coefs'] = decoding_coefs
    nclasses = decoding_coefs.shape[0]
    ntrials = nclasses * 150
    perturbed_states, perturbed_classes = np.zeros((args.nperturb,ntrials,old_Xs.shape[1])),np.zeros((args.nperturb,ntrials))
    perturbed_rewards,perturbed_proj_coefs = np.zeros((args.nperturb,ntrials)),np.zeros((args.nperturb,nclasses,ntrials))
    for i in range(args.nperturb):
        states,classes,rews = apply_perturbation(setup_run_dict,perturb_dict)  
        states = states.reshape(ntrials,old_Xs.shape[1])
        proj_coefs = decoding_coefs @ states.T
        perturbed_states[i] = states #.reshape(ntrials,old_Xs.shape[1])
        perturbed_classes[i] = classes.reshape(ntrials)
        perturbed_rewards[i] = rews.reshape(ntrials)
        perturbed_proj_coefs[i] = proj_coefs
    perturb_trial_data_dict = {'states':perturbed_states, 'proj_coefs':perturbed_proj_coefs,
                            'classes':perturbed_classes,'rewards':perturbed_rewards,'clf':clf,
                            'perturb_dict':perturb_dict}
    return perturb_trial_data_dict

def get_change_in_classification(index,perturb_trial_data_dict,t_before,t_after):
    '''
    Calculates the difference in classification accuracy during a time window before and after perturbation
    returns this and the classification over time after perturbation (to plot curves)
    Arguments:
        perturb_trial_data_dict (dict): contains data from perturbation trials
        t_before,t_after (int): specifies how many trials before and after the perturbation to save
    Returns:
        acc_before, acc_after np.array(floats): accuracy of the linear classifier around the time of perturbation
    '''
    states = perturb_trial_data_dict['states'][index]
    classes = perturb_trial_data_dict['classes'][index]
    clf = perturb_trial_data_dict['clf']
    perturb_dict = perturb_trial_data_dict['perturb_dict']
    acc = predict_opponent_class(states,classes,clf)
    return get_before_after(perturb_dict,acc,t_before,t_after)

def get_before_after(perturb_dict,data,t_before,t_after):
    t = perturb_dict['time_of_perturbation']+150*perturb_dict['opponent_index_to_perturb_from'] 
    data_before = data[t-t_before+1:t+1] #correcting an off-by-one error for the sake of plotting   
    data_after = data[t+1:t+1+t_after]
    return data_before,data_after

def get_change_in_reward(index,perturb_trial_data_dict,t_before,t_after):
    '''
    Calculates the average reward in a time window immediately before the perturbation
    and compares that the to the average reward in a time window immediately after
    This only does it for a single perturbation but another function will aggregate
    Returns this value and the rewards in the rest of the block after perturbation
    Same arguments and return essentially as get_change_inc_classification
    '''
    rews = perturb_trial_data_dict['rewards'][index]
    perturb_dict = perturb_trial_data_dict['perturb_dict']
    return get_before_after(perturb_dict,rews,t_before,t_after)

def divide_perturbations_into_before_after(args,perturb_trial_data_dict):
    '''
    Given the list of perturbation trial data in the dictionary, splits the data into the time before and after
    the perturbation and saves reward and classifier accuracy
    Arguments:
        args (ArgumentParser): command line arguments
            tbefore,tafter (int): specifies how many trials before and after the perturbation to save
        perturb_trial_data_dict (dict): perturbation trial data
    Returns:
        rew_before,rew_after,clf_acc_before,clf_acc_after (np.arrays): reward of agent before after peturbation
                                linear decoder accuracy before and after perturbation
    '''
    rew_before,clf_acc_before = np.zeros((args.nperturb,args.tbefore)), np.zeros((args.nperturb,args.tbefore))
    rew_after,clf_acc_after = np.zeros((args.nperturb,args.tafter)), np.zeros((args.nperturb,args.tafter))
    for i in range(args.nperturb):
        before,after = get_change_in_classification(i,perturb_trial_data_dict,args.tbefore,args.tafter)
        clf_acc_before[i] = before
        clf_acc_after[i] = after

        before,after = get_change_in_reward(i,perturb_trial_data_dict,args.tbefore,args.tafter)
        rew_before[i] = before
        rew_after[i] = after
    return rew_before,rew_after,clf_acc_before,clf_acc_after

def plot_perturbation_result(rew_before,rew_after,clf_acc_before,clf_acc_after,sigma=2):
    '''
    Concatenates before and after, gets mean and stderr of the mean over time, smooths, and plots!
    '''
    nperturb,t_before = rew_before.shape
    t_after = rew_after.shape[1]
    plt.figure(figsize=(5,4))
    cm = plt.get_cmap('Set2').colors

    def avg_std(data):
        mu = data.mean(axis=0) #data has shape (nperturb, time)
        stderr = data.std(axis=0)/np.sqrt(nperturb)
        mu = gaussian_filter(mu,sigma=sigma)
        return mu,stderr

    def plot_before_after(before,after,name,cm):
        mu_before,stderr_before = avg_std(before)
        mu_after,stderr_after = avg_std(after)
        mu = np.concatenate([mu_before,mu_after])
        stderr = np.concatenate([stderr_before,stderr_after])

        x = np.arange(-t_before,t_after)
        plt.plot(x,mu,label=name,color=cm)
        plot_errorbar(x,mu,stderr,color=cm)
    
    plot_before_after(rew_before,rew_after,"Reward",cm[0])
    plot_before_after(clf_acc_before,clf_acc_after,"Classification",cm[1])

    plt.xlabel("Time from Perturbation")
    plt.ylabel("P(correct)")
    plt.axvline(x=0,c="black")
    plt.legend(fontsize=10,loc="lower right",frameon=True)
    plt.tight_layout()
    plt.savefig(Path("results","Perturbation.pdf"))

if __name__ == "__main__":
    parser = default_argparser("Peturbation analysis on top of linear classifier")
    parser.add_argument("--nperturb",required=False,type=int,default=15, help = "number of perturbation blocks")

    parser.add_argument("--tbefore",required=False,type=int,default=15, help = "pre-perturbation trials to save")
    parser.add_argument("--tafter",required=False,type=int,default=15, help = "post-perturbation trials to save")
    parser.add_argument("--sigma",required=False,type=int,default=2, help = "smoothing param for plotting")

    args = parser.parse_args()   
    #get the run parameters
    model,_ = get_net(args.trun,args.mtype,args.trainiters)  
    with open(Path(FILEPATH,"data","models",f"run{args.trun}",Path("train_params.json") ) ) as f:
        script_kwargs = json.load(f)
    train_params = convert_dist_to_params(script_kwargs)
    env = get_env_from_json(train_params['env'],useparams=True) 
    test_opponents = define_decoding_test_opponents()
    #save in a dictionary for efficiency
    setup_run_dict = {'model':model,'train_params':train_params,'env':env,'test_opponents':test_opponents}

    perturb_dict = get_perturb_from_json()
    perturb_trial_data_dict = run_multiple_perturb_trials(args,setup_run_dict,perturb_dict)
    rew_before,rew_after,clf_acc_before,clf_acc_after = divide_perturbations_into_before_after(args,perturb_trial_data_dict)
    plot_perturbation_result(rew_before,rew_after,clf_acc_before,clf_acc_after,args.sigma)
