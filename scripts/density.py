from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import kde
from scipy.special import kl_div
from sklearn.decomposition import PCA

from mprnn.utils import set_plotting_params,get_env_and_test_opps, FILEPATH, default_argparser
from mprnn.representation import get_repvecs
from mprnn.repscomparison import load_repscomparison,RepsComparison
from pca import get_states_opps
set_plotting_params()

def get_repvecs_for_density(model,reps,opponent_inds,args):
    '''
    sample the repvecs more (more iterations) to do density estimation
    '''
    env,_ = get_env_and_test_opps(args.runindex)
    state_reps = []
    for _ in range(args.nblocks): #losing the order because of this but we are already specifying opponents, so it's fine
        for i,opp in enumerate(reps.opponent_order):
            if i not in opponent_inds:
                continue      
            print("opponent index", i)
            if args.userepvecsfordensity:
                _,state_rep = get_repvecs(env,model,opp,1,args.seqlength) #only 1 run, we don't want to average
            else:
                state_rep = get_states_opps(reps,opponent_inds,64) #32 individual trajectories instead, more data but maybe more representatitve
            state_reps.append(state_rep)
    state_reps = np.vstack(state_reps)
    return state_reps

def get_states_dict(model,reps,args):
    '''
    Gets a sample of the states for each opponent class type (defined by reps.index_dict) and saves
    in a dictionary with the same structure
    '''
    states_dict = {}
    for name,opponent_inds in reps.index_dict.items():
        states_dict[name] = get_repvecs_for_density(model,reps,opponent_inds,args)
    return states_dict

def reduce_all_opponents(states_dict,args):
    '''
    Given a dictionary of states, fit to the wd opponents and project all the other ones
    Arguments:
        states_dict (dict): keys are names of ood opponent type, values are np.arrays of states
    returns
        reduced_dict(dict): same strucutre, this time they are reduced dimensionality to 10
    '''
    reduced_dict = {}
    pca_wd = PCA(n_components=args.npcs).fit(states_dict['within-dist']) 
    for name in states_dict.keys():
        reduced_dict[name] = pca_wd.transform(states_dict[name])
    return reduced_dict

def estimate_density(reduced_dict,args):
    '''
    Given the reduced npc figure, estimates the density on the two pc's specified on args
    only doing 2 pcs because the grid grows exponentially, so it is computationally intractable to go larger
    It's also easier to visualize in 2d
    Arguments:
        reduced_dict(dict): keys(opponent classes), values(reduced representations), this time they are reduced dimensionality to 10
        args (ArgumentParser)
    Returns:
        density_dict (dict): same structure as reduced_dict, but now with densities over a 2d grid
        xi,yi (np.arrays): defining the grid
        binarea (float): the area of each cell in the grid
    '''
    #define the bins over the entire dataset
    allr = np.concatenate(list(reduced_dict.values()))
    x,y = allr[:,args.xpc],allr[:,args.ypc]
    xi, yi = np.mgrid[x.min():x.max():args.nbins*1j, y.min():y.max():args.nbins*1j]
    binarea = (x.max()-x.min())*(y.max()-y.min())/(args.nbins**2)
    density_dict= dict()
    for name,reduced in reduced_dict.items():
        x,y = reduced[:,args.xpc],reduced[:,args.ypc]
        data = [x,y]
        k = kde.gaussian_kde(data)
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        density_dict[name] = zi
    return density_dict,binarea,xi,yi

def plot_densities(density_dict,xi,yi,args):
    '''
    Plots the densities in a heatmap
    Arguments:
        density_dict (dict): keys are differnet type of opponent classes, values are density np.array on xy grid
        xi,yi (np.arrays): the x and y values on that grid
        args (ArgumentParser)
    '''
    zimax = np.max(density_dict['within-dist']) #use the WD dist to set the range
    plt.rcParams['image.cmap'] = 'plasma'
    fig,ax = plt.subplots(len(density_dict.keys())//2,2,figsize=(14,7),sharex='all',sharey='all') #n x 2 grid
    for i,(name,density) in enumerate(density_dict.items()):  
        a,b = i % 2, i // 2
        pc = ax[a,b].pcolormesh(xi, yi, density.reshape(xi.shape), shading='auto',vmin = 0, vmax = zimax)
        ax[a,b].set_title(name)

        fig.colorbar(pc,ax=ax[a,b])
        
    fig.savefig(Path(FILEPATH,"results",f"KDEvisPC{args.xpc}{args.ypc}.pdf"))
    plt.show()

def plot_relative_densities(density_dict,binarea):
    '''
    identify relative similarity of estimated densities of the wd dist and the others
    get the cumulative density of the distributions that are not on the wd support
    plot a line plot and save in results
    Arguments:
        density_dict (dict): keys are the different type of opponents, values are np.arrays of density on an xy grid
        binarea (float): relative area of the bins on that xy grid
    '''
    ds = np.arange(0.00001,max(np.concatenate(list(density_dict.values())))+0.01,0.00001)
    plt.figure(figsize=(7,4))
    wd_density = density_dict['within-dist']
    for name,density in density_dict.items():
        print("KL-Div: ", name," | WD", np.sum(kl_div(density,wd_density)))
        d = 7e-3

        rare = np.sum(density[np.logical_and(wd_density < d,wd_density >= 0)]*binarea)
        print(f"% Covered by WD support of density < {d}: ", rare*100)
        
        covered = np.zeros((len(ds)))
        for j,d in enumerate(ds):
            covered[j] = np.sum(density[np.logical_and(wd_density < d,wd_density >= 0)]*binarea)
        plt.plot(ds,covered,label=name)
        near50 = np.argmin(np.abs(covered - 0.5))
        plt.axvline(ds[near50],alpha=1.0,lw=0.5)
    plt.legend()
    plt.xscale('log')
    plt.xlabel("d")
    plt.ylabel("CDF ($D_{state}$ < d)")
    near10 = np.argmin(np.abs(covered - 0.1))
    plt.axvline(ds[near10],c="black",alpha=1.0,lw=0.5)
    plt.savefig(Path(FILEPATH,"results","cdfKDE.pdf"))

if __name__ == "__main__":
    parser = default_argparser("Loads saved representation from rsa analysis, does pca on different ood classes and plots")
    parser.add_argument("--nbins",default=100,type=int,required=False,help="number of bins for each dimension of density")
    parser.add_argument("--xpc",default=0,required=False,type=int,help="PC index for the x axis of density plot")
    parser.add_argument("--ypc",default=1,required=False,type=int,help = "PC index for the y axis of density plot")
    parser.add_argument('--userepvecsfordensity', action='store_true',help="use the fixed-sequence reps for density estimatino")
    parser.add_argument('--usestatesfordensity', dest='feature', action='store_false',help="use real trajectories for density estimation")

    #here, use loadrepscomparison path if we've already calculated these in scatter_distances.py
    #it is very time intensive as it has to be exhaustive so only run that once
    args = parser.parse_args()
    reps = load_repscomparison(args)
    states_dict = get_states_dict(reps.model,reps,args)
    reduced_dict = reduce_all_opponents(states_dict,args)

    density_dict,binarea,xi,yi = estimate_density(reduced_dict,args)
    plot_densities(density_dict,xi,yi,args)
    plot_relative_densities(density_dict,binarea)