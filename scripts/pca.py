from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mprnn.testing import test_specific_opponents
from mprnn.utils import set_plotting_params, FILEPATH,get_env_and_test_opps,default_argparser,get_net
from mprnn.repscomparison import load_repscomparison,RepsComparison
set_plotting_params()

def calculate_var_explained(states,pca):
    '''
    Given a np.array of representation states and a fitted pca model,
    calculates the variance explained by that model on the states
    Arguments:
        states (np.array): hidden states from the model, maybe averaged
        pca (PCA): pca model
    returns:
        float (variance explained of states by the pca components)
    '''
    #de-mean states
    states -= np.mean(states,axis=0)
    components = pca.components_
    reduced = states @ components.T
    reconstruction = reduced @ components
    error = np.sum((states - reconstruction)**2) #np.linalg.norm includes the sqrt
    relerr = error / np.sum(states**2)
    return np.abs(1-relerr)

def get_states_opps(reps,opponent_inds,nblocks=0):
    '''
    Arguments:
        reps (RepsComparison): reps to get the opponent order, and save in the same order
        opponent_inds (list): list of indices of opponents in reps.opponent_order
    '''
    env,_ = get_env_and_test_opps(reps.args.runindex)
    trialdata = []
    if nblocks == 0:
        nblocks = reps.args.nblocks
    for i,opp in enumerate(reps.opponent_order):
        if i not in opponent_inds:
            continue
        print("testing opponent",i)
        for _ in range(nblocks):
            prewashout, postwashout = test_specific_opponents(env, reps.model, [opp],nwashout = reps.args.nwashout)
            trialdata.extend(prewashout) #actions,rewards,states,preds,opps,opp_inds,opps_kwargs
            trialdata.extend(postwashout)
    states = np.vstack([trial.states for trial in trialdata])

    return states

def get_var_explained(reps,opponent_inds):
    '''
    Get a sample of variance explained for different dimensions of pca models
    Gets a sample of states nrun times, fits a model of size 1:npcs on the wd opponents, then saves
    Arguments:
        reps (RepsComparisonObject)
        opponent_inds (the opponents to use for the test variance explained).
        in reps:
            args.npcs (int): max number of pc's to test
            args.nruns (int): how many samples of the var explained to get
    '''
    var_exps = np.zeros((reps.args.nblocks,reps.args.npcs))
    for j in range(reps.args.nblocks):
        states = get_states_opps(reps,opponent_inds)
        states_wd_opps = get_states_opps(reps,reps.wd_inds) #use this to fit the pca
        for i in range(reps.args.npcs):
            pca_wd = PCA(n_components=i+1).fit(states_wd_opps) #in theory we could just do this once, but it's a fast operation
            #I'm also not sure if pca.components_ is sorted
            var_exps[j,i] = calculate_var_explained(states,pca_wd)
    return var_exps
    
def get_var_explained_dict(reps):
    '''
    Gets the variance explained curve from get_var_explained for all different ood opponent types (and wd)
    '''
    varexps_dict = dict()
    for name,opponent_inds in reps.index_dict.items():
        varexps_dict[name] = get_var_explained(reps,opponent_inds)
    return varexps_dict

def plot_pcs(varexps_dict):
    '''
    Plots the variance explained curve, separted by ood oponent type (and wd), then saves in Results folder
    '''
    x = range(1,11)
    plt.figure(figsize=(8,6))
    for n,v in varexps_dict.items():
        mexp = np.mean(v,axis=0)
        stdexp = np.std(v,axis=0)
        plt.plot(x,mexp,label=n)
        plt.fill_between(x,mexp-stdexp,mexp+stdexp,alpha=0.4)
    plt.legend()
    plt.xlabel("PCA component")
    plt.ylabel("Cumulative Variance Explained")
    plt.axhline(0.95)
    plt.savefig(Path(FILEPATH,"results","PCAcomponents.pdf"))

if __name__ == "__main__":
    parser = default_argparser("Loads saved representation from rsa analysis, does pca on different ood classes and plots")
    args = parser.parse_args()
    reps = load_repscomparison(args)
    varexps_dict = get_var_explained_dict(reps)
    plot_pcs(varexps_dict)