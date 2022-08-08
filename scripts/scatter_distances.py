import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mprnn.utils import BETTER_NAMES,FILEPATH,default_argparser,set_plotting_params
from mprnn.repscomparison import load_repscomparison,RepsComparison
set_plotting_params()

def scatter_distances(reps,args):
    '''
    All we are doing in this file is loading the saved repscomparison object, and plotting 
    scatter plots of the distances. 
    This makes two plots, one is the min vs average distance of WD and OOD, which has all opponents
    The other one is the min vs LC mapping distance of WD and OOD opponents, which only has those that are good LC fits.
    The second one is the one that went into the figure
    '''
    cutoffdist = args.cutoffdist

    _,ax = plt.subplots(1,2,sharey = 'col',figsize=(12,5))
    ood_opps = np.array([o[0] for o in reps.ood_opponents])

    colors = ["#B0FE76","#06C1AC","navy","darkolivegreen","#025EF2","coral","#DA1B5B","#990D35"]

    goodLCmap = reps.LCmap_scores > args.cutoffscore
    for j in range(2):
        for i,opp in enumerate(np.unique(ood_opps)):
            inds = ood_opps == opp
            if j:
                xlabel = "min distance from ood to LC mapping"
                y = reps.mindists[np.logical_and(goodLCmap,inds)]
                x = reps.LCmap_dists[np.logical_and(goodLCmap,inds)]
            else:
                xlabel = "avg distance from ood to wd"
                y = reps.mindists[inds]
                x = reps.avgdists[inds]
            if len(y) == 0:
                continue
            ax[j].scatter(x,y,label=BETTER_NAMES[opp],s=20,color=colors[i])
            
        ax[j].set_ylabel("min distance from ood to wd")
        ax[j].set_xlabel(xlabel)
    ax[0].legend()
    ax[0].axline((0.9, 0.9), slope=1,c="black") #x=y line
    ax[1].axline((0.9, 0.9), slope=1,c="black")
    ax[1].legend()

    xmin,xmax = 0,5.5
    ax[1].set_xlim([xmin,xmax])
    ax[0].set_xlim([3.75,7.5])
    ymin = 0.25 #(cutoffdist-xmin)/(xmax-xmin)
    ax[1].axvline(cutoffdist,ymin=ymin,c="orange")
    ax[1].axhline(cutoffdist)
    ax[1].axhspan(0,cutoffdist,alpha=0.3,label="Template Matching")
    ax[1].axvspan(0,cutoffdist,ymin=ymin,alpha=0.1,color="orange",label="Interpolation")
    ax[0].axhspan(0,cutoffdist,alpha=0.3)
    ax[0].axhline(cutoffdist)

    plt.savefig(Path(FILEPATH,"results","MinandLCdistscatter.pdf",dpi=300))

if __name__ == "__main__":
    parser = default_argparser("Loads saved representation from rsa analysis and plots separation by ood type")

    args = parser.parse_args()
    args.seqlength *= 2 #the seqlength in input corresponds to the number of steps to feed, but the dimension is 2, so
    #we need to iterate over 2**(2*seqlength) binary combinations
    reps = load_repscomparison(args)
    if args.savedrepscomparisonpath is None:
        reps.save()
    #save the model here, as this is the first of scatter_distances,pca,density files that will all load it
    scatter_distances(reps,args)