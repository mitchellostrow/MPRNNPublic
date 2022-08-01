import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats,spatial
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch

from mprnn.utils import default_argparser, FILEPATH, BETTER_NAMES
from mprnn.representation import save_repvecs_across_opponents

def compare_populations(pop1,pop2,simfunc):
    '''
    this functions works on either policy_reps or state_reps
    '''
    assert pop1.shape == pop2.shape
    if len(pop1.shape) == 3:
        pop1 = pop1.reshape(pop1.shape[0],-1) #flatten along the final dimensions
        pop2 = pop2.reshape(pop2.shape[0],-1)
    avg_similarity = 0
    n = pop1.shape[0]**2
    for rep1 in pop1:
        for rep2 in pop2:
            avg_similarity += 1/n * simfunc(rep1,rep2)
    return avg_similarity

def compare_reps(representations,opponent_order,simfunc="pearsonr"):
    '''
    Compares the representations for both policy and state opponent by opponent, comparing 
    populations along the way
    Returns:
        policy_similarities,state_similarities (np.arrays): opponent x opponent average population similarity
    '''
    if simfunc == "pearsonr":
        simfunc = lambda x,y:stats.pearsonr(x,y)[0] #just want r for now, not p value
    elif simfunc == "cosine":
        simfunc = lambda x,y: spatial.distance.cosine(x,y)
    num_opps = len(opponent_order)
    similarities = np.zeros((num_opps,num_opps))
    for i in range(num_opps):
        for j in range(num_opps):
            similarities[i,j] = compare_populations(representations[i],representations[j],simfunc)

    return similarities

def get_avg_similarites(similarities,opponent_order):
    '''
    Given population similarities, average over opponent types to get an average
    opponent class vs opponent_class similarity
    '''
    unique_opponents = np.unique(opponent_order)
    nopp = len(unique_opponents)
    avg_similarities = np.zeros((nopp,nopp))

    def get_mean(o1,o2,data):
        data_i = data[opponent_order==o1]
        data_ij = data_i[:,opponent_order==o2]
        mu = data_ij.mean()
        return mu
    #average across opponents and plot this comparison too
    for i,o1 in enumerate(unique_opponents):
        for j,o2 in enumerate(unique_opponents):
            avg_similarities[i,j] = get_mean(o1,o2,similarities)

    return avg_similarities,unique_opponents

def plot_state_vector_lowd(state_reps,opponent_order):
    '''
    Runs the t-sne and plots the state representations for all opponents, averaged over sequences and populations
    doesn't return anything, but saves the plot
    '''

    tsne = TSNE()
    state_reps = [s.mean(axis=0) for s in state_reps] #average over the populations
    all_states_avged_overseq = np.array([s.mean(axis=0) for s in state_reps]) #average over the sequence too
    #now we just have a single 128 dimensional vector for the average state, the center of the representation
    reduced = tsne.fit_transform(all_states_avged_overseq)
    for j in range(2):
        _, ax = plt.subplots(1,1,figsize=(10,10))
        plt.set_cmap('Pastel1')
        for opponent in (np.unique(opponent_order)):
            reduced_opp = reduced[opponent_order==opponent]
            if opponent in {'all','reversalbandit','softmaxqlearn','epsilonqlearn','mimicry'}:
                marker = "*"
            else:
                marker = "."

            if j == 0:
                label = ("OOD","orange") if marker == "*" else ("WD","blue")
                plt.scatter(reduced_opp[:,0],reduced_opp[:,1],label=label[0],s=160,marker=marker,color=label[1])
            else:
                label = BETTER_NAMES[opponent]
                plt.scatter(reduced_opp[:,0],reduced_opp[:,1],label=label,s=160,marker=marker)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)    
        ax.set_xlabel("t-SNE 1",fontsize=30)
        ax.set_ylabel("t-SNE 2",fontsize=30)
        ax.set_title("Recurrent State low-d visualization",fontsize=30)
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticklabels([])

        handles, labels = plt.gca().get_legend_handles_labels() #remove repeats from the legend
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),fontsize=28)
        plt.tight_layout()
        plt.savefig(Path(FILEPATH,"results",f"reduced_tsne{j}.pdf"))     

def plot_rsa(similarities,opponent_order,name,sorted=False):
    '''
    Given the similarities calculated before and the list of opponents,
    plot the RSA as a heatmap, arranged into hierarchical groups or not
    This will plot opponent by opponent rsa
    '''
    idx1 = np.array([np.where(opponent_order==i)[0] for i in BETTER_NAMES.keys()])
    idx1 = idx1.flatten()
    # #sort the matrix
    if sorted:
        Y = sch.linkage(similarities, method='centroid')
        Z = sch.dendrogram(Y, orientation='left')
        idx1 = Z['leaves']
    # #dendrogram plots a figure so we need to set the figure here
    fig, ax = plt.subplots(1,1,figsize=(6.75,6.5))

    #sort both dimensions of the matrix
    similarities = similarities[idx1,:]
    similarities = np.array(similarities[:,idx1])
    opponent_order = opponent_order[idx1]
  
    cax = ax.matshow(similarities,cmap='plasma')   
    cax.set_clim(-0.3,1)   
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    if len(opponent_order) < 12: #if we'll be able to read them, label the opponents
        labels = []

        labels.extend([BETTER_NAMES[i] for i in opponent_order])
        ax.xaxis.set_ticks([i for i in range(len(labels))])
        ax.yaxis.set_ticks([i for i in range(len(labels))])

        ax.set_xticklabels(labels,fontsize=12,rotation=65)
        plt.gca().xaxis.tick_bottom() #move xlabel to the bottom
        ax.set_yticklabels(labels,fontsize=12)
    if name[:6] == "policy":
        plt.title("Policy RSA", fontsize=20)
    else:
        plt.title("Recurrent state RSA",fontsize=20)
    plt.tight_layout()
    plt.savefig(Path(FILEPATH,"results",f"RSA_{name}.pdf"))
    plt.close()

    return idx1

if __name__ == "__main__":
    parser = default_argparser("get and plot representational similarity analysis")
    parser.add_argument('--seqlength',default=3, required = False,type=int, help = "number of time steps for the fixed sequence")
    parser.add_argument("--npop",default=1,required=False,type=int,help = "number of times to get the representation vecotrs")
    parser.add_argument("--sorted",default=False,required=False,type=bool,help="whether or not to sort the RSA plot")
    args = parser.parse_args()   

    policy_reps,state_reps,opponent_order = save_repvecs_across_opponents(args)
    policy_similarities = compare_reps(policy_reps,opponent_order)
    state_similarities = compare_reps(state_reps,opponent_order)
    avg_state_similarities,unique_opponents = get_avg_similarites(state_similarities,opponent_order)
    avg_policy_similarities,_ = get_avg_similarites(policy_similarities,opponent_order)

    plot_state_vector_lowd(state_reps,opponent_order)
    plot_rsa(avg_state_similarities,unique_opponents,"state_across_opponents",sorted=args.sorted)
    plot_rsa(avg_policy_similarities,unique_opponents,"policy_across_opponents",sorted=args.sorted)