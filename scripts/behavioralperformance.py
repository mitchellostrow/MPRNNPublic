from mprnn.compare import make_dev_from_baseline
from mprnn.testing import gather_data_time
from mprnn.utils import set_plotting_params
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def get_reward_deviations(blockdata,baseline=None):
    '''
    Given a dataset TrialTestObject with data saved from a block,trial, or run, calculates the reward deviation of the agent
    on all opponents against a baseline. If baseline is unspecified, will use the single-task trained agent baseline saved in
    models/baselines/baselines.p
    Arguments:
        blockdata (TrialTestObject): data from the trial
        baseline (TrialTestObject): data from a baseline trial on which to compare
    Returns:
        df_reshaped (pd.DataFrame): dataframe with samples, 
            on each row there is the deviation of reward from baseline (strict deviation, not z-score) and opponent name.
            
    '''
    deviations_by_opp = make_dev_from_baseline(blockdata,baseline=baseline)
    #separate out by opponent and get the names from the data
    df = pd.DataFrame() #compile reward distribution into a dataframe
    l = min([len(o) for o in deviations_by_opp.values()]) #because the dataframe has to be square
    for opponent_name, avgdev in deviations_by_opp.items():
        df[opponent_name] = avgdev[:l]
    #going to ignore the default agent for this stuff because it doesn't really matter
    df_reshaped = pd.DataFrame()
    df_reshaped['deviation'] = df.to_numpy().T.flatten()
    BETTER_NAMES = {'MP 1st+2nd': 'MP1+2','anti-correlated':'AB','Epsilon-Greedy QL':'$\epsilon$-$\\bf{QL}$',
                   'Softmax QL':'S-QL','n-back mimickry':'MC','Linear Combination':'LC','MP 1st':'MP1','Patterns':'PB'}
    
    opponents = [BETTER_NAMES[o] for o in deviations_by_opp.keys() for _ in range(len(df))]
    df_reshaped['opponent'] = opponents
    
    return df_reshaped

def get_within_ood_deviations(blockdata_within,blockdata_ood,baseline_within=None,baseline_ood=None):
    '''
    Given some block data for both within-distribution and out of distribution opponents, along with potential baselines,
    Runs the above function for each to extract the relative deviations and combines them.
    Arguments:
        blockdata_within (TrialTestObject): block data saved from within-distribution opponents
        blockdata_ood (TrialTestObject): block data saved from out-of-distribution opponents
        baseline_within (TrialTestObject): block data saved from within-d baselines (in get_trained_untrained, the baseline is untrained)
        baseline_ood (TrialTestObject): block data saved from ood baseline
    Returns:
        total (pd.DataFrame): dataframe with deviations for all opponents
    '''
    within_reshaped = get_reward_deviations(blockdata_within,baseline=baseline_within)
    ood_reshaped = get_reward_deviations(blockdata_ood,baseline=baseline_ood)
    total = within_reshaped.append(ood_reshaped)
    return total

def get_trained_untrained_dataframes(args):
    '''
    Gathers data from the trained and untrained opponents specified in the arguments (parsed from the script),
    calculates the deviations of the trained multi-task opponent against the trained baseline (pre-saved)
    and the untrained (given in script), then concatenates the dataframes
    Inputs:
        args (ArgumentParser object): contains info on which trained run to use, model type, and trained runtime for testing
    Outputs:
        totaldf (pd.DataFrame): dataframe combining deviations and labels for opponent and baseline type for all opponents and baselines.
    '''
    blockdata_within,blockdata_ood = gather_data_time(args.trun,args.mtype,args.trainiters,args.nblocks)
    blockdata_within_untrained,blockdata_ood_untrained = gather_data_time(args.urun,args.mtype,"0k",args.nblocks)

    total_trained = get_within_ood_deviations(blockdata_within,blockdata_ood)
    total_untrained = get_within_ood_deviations(blockdata_within,blockdata_ood,baseline_within=blockdata_within_untrained,
                                baseline_ood=blockdata_ood_untrained)
    total_trained["baseline"] = ["Multi-task trained - Single-task trained"]*len(total_trained)
    total_untrained["baseline"] = ["Multi-task trained - Untrained"]*len(total_untrained)
    totaldf = total_untrained.append(total_trained)
    return totaldf

def plot_violins(totaldf):
    ''''
    Plots a categorical violinplot of deviation from baseline reward for all given opponents over all blocks
    Arguments:
        totaldf (pd.DataFrame): output from the previous function
    Returns:
        None
    '''
    set_plotting_params()
    plt.rcParams["font.size"] = 27
    _, ax1 = plt.subplots(1, 1, sharex=True,figsize=(14,8))
    plt.rc('ytick', labelsize='medium')
    # plot the same data on both axes
    sns.violinplot(ax=ax1,x="opponent",y="deviation",data=totaldf,hue="baseline",width=0.5)

    plt.xticks(rotation=0,fontsize=30)
    ax1.set_xlabel('')
    ax1.set_ylabel("$\mu_{multi}-\mu_{baseline}$",fontsize=35)
    ax1.axhline(y=0,c="black") #plot x-axis
    ax1.axvspan(-0.2,2.5,color="green",alpha=0.1) #color the first 3 opponents as within-distribution
    ax1.axvline(x=2.5,alpha=0.1) #separating line between wd and ood
    ax1.axvspan(2.5, 7.5, alpha=0.1)
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig(Path("results","RewardDeviation.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and Test the Agent's Performance on all \
                                                    8 opponents relative to a trained and untrained baseline")

    parser.add_argument("--trun", type=int, required=False, default=86, help="model run # of the \
                                                                        trained multi-task agent")
    parser.add_argument("--urun", type=int, required=False, default=6, help="model run # of the \
                                                                     untrained multi-task agent")
    parser.add_argument("--mtype", required=False, default = "SSP", help="type of model (SSP or notSSP)")
    
    parser.add_argument("--trainiters", required=False, default="8000k", help="length of model training to be used (number+k)")
    parser.add_argument("--nblocks",type=int, required=False, default="50", help="number of trial blocks to test each opponent on")

    args = parser.parse_args()
    
    totaldf = get_trained_untrained_dataframes(args)
    plot_violins(totaldf)