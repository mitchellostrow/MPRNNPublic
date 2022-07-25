import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from scipy.ndimage import gaussian_filter
from copy import deepcopy

from mprnn.testing import test_specific_opponents
from mprnn.utils import get_net, set_plotting_params, get_env_and_test_opps,iterate_over_opponents,convert_names_short

set_plotting_params()

def get_all_opponent_params(opponent_pairs,test_opponents):
    '''
    Given a list of list of opponent pairs and the dictionary of parameters, returns all possible opponents
    Arguments:
        opponent_pairs (list(list)): list of lists of opponent pairs on which to test switching between
        test_opponents (dict): from the environment loading, the type of opponents and parameters
    Returns:
        opponents (list(tuple(str,dict)): details every possible opponent, (opponent name, opponent_parameters)
    '''
    opponents = [[],[]]
    for i in range(len(opponent_pairs)):
        for opp,param in iterate_over_opponents(test_opponents):
            if opp in opponent_pairs[i]:
                opponents[i].append((opp,param))
    return opponents

def get_switching_behavior(prewashout_trials,postwashout_trials,opponent_pairs,
                            preswitch=20,postswitch=30,block_length=150):
    '''
    Given some data in the form of lists of TrialTestObjects, parse it by saving only 
    the average reward and std err of the reward before and after switching between the two opponents
    Arguments:
        prewashout_trials: list(list(TrialTestObject)): first list indexes the opponent_pairs, secon the numb
    '''
    numblocks = len(prewashout_trials[0]) 
    save_length = preswitch+postswitch #how many trial times to save in total?
    switch_periods = np.zeros((len(opponent_pairs),numblocks*2,save_length))
    #prewashout and post_washout are the same length, so total number of blocks is *2
    avg_switches = []
    upper_bound_stderrs = []
    lower_bound_stderrs = []

    for opp in range(len(opponent_pairs)):
        for i in range(0,numblocks): 
            data = prewashout_trials[opp][i],postwashout_trials[opp][i]
            for j,block in enumerate(data): #iterate over pre and postwashout, save separately
                before_switch_rew = block.rewards[block_length-preswitch:block_length]
                after_switch_rew = block.rewards[0:postswitch]#similarly, 0:30 is right after switch
                #stack these two into a block
                index = i+j*numblocks #stack the postwashout trials at the end (j=1)
                switch_periods[opp,index,0:preswitch] = gaussian_filter(before_switch_rew,sigma=1) 
                #filtering is optional, but change manually
                switch_periods[opp,index,preswitch:save_length] = gaussian_filter(after_switch_rew,sigma=1)
        #average our switches over all trials
        avg_switch = np.mean(switch_periods[opp],axis=0) #average over all switch blocks
        std_switch = np.std(switch_periods[opp],axis=0)
        upper = avg_switch + std_switch
        lower = avg_switch - std_switch
        
        avg_switches.append(avg_switch)
        upper_bound_stderrs.append(upper)
        lower_bound_stderrs.append(lower)

    return avg_switches,upper_bound_stderrs,lower_bound_stderrs

def plot_switching_behavior(opponent_pairs,avg_switches,upper_bound_stderrs,lower_bound_stderrs,
                           preswitch=20,postswitch=30):
    '''
    Given the arrays of data to plot, makes the time-varying reward plot with the switch at time preswitch
    Arguments:
        opponent_pairs (list(list(str)): list of lists of strings that define the test opponent names
        avg_switches (list(list(float))):
        upper_bound_stderrs (list(list(float))):
        lower_bound_stderrs (list(list(float))):
        preswitch (int): how many trials are saved before the switch?
        postswitch (int): how many trials are saved after the switch?
    Returns:
        None
    '''
    save_length = preswitch+postswitch 
    _, axes = plt.subplots(1,len(opponent_pairs),figsize=(6*len(opponent_pairs),5),sharey=True)
    for i,(avg_switch,upper,lower) in enumerate(zip(avg_switches,upper_bound_stderrs,lower_bound_stderrs)):
        
        #black line at switch time
        axes[i].plot(np.arange(-1,1),avg_switch[preswitch-1:preswitch+1],c="black",marker="*")
        #preswitch behavior
        axes[i].plot(np.arange(-preswitch,0),avg_switch[0:preswitch],label="Opponent A")
        #postswitch behavior
        axes[i].plot(np.arange(0,postswitch),avg_switch[preswitch:save_length],label="Opponent B")

        #error bars
        axes[i].plot(np.arange(-preswitch,0), lower[0:preswitch], color='tab:blue', alpha=0.1)
        axes[i].plot(np.arange(0,postswitch),upper[preswitch:save_length], color='tab:blue', alpha=0.1)
        axes[i].fill_between(np.arange(-preswitch,postswitch), lower, upper, alpha=0.2)

        axes[i].set_ylim(0.2,1.1)
        axes[i].set_xlabel("Time relative to switch")
        #make pretty labels
        l = convert_names_short(opponent_pairs[i][0]) + " + " + convert_names_short(opponent_pairs[i][1])
        axes[i].set_title(l)

    axes[0].set_ylabel("Avg Reward",fontsize=30)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path("results","SwitchBehavior.pdf"))

def test_opponents_switch(args,opponent_pairs):
    '''
    Given input arguments to the script and the parsed opponent paris, test the opponents and save data, then analyze and plot
    Arguments:
        args (ArgParse): arguments from script:
            trun: model run index of the trained agent
            mtype: whether the model is SSP or default A2C
            trainiters: how long the model has traned for
            nwashout, ntests: how long the model should be tested + washed out for
        opponent_pairs (list(list(str))): list of opponent pairs to be tested on switching
    Returns:
        None
    '''
    model,_ = get_net(args.trun,args.mtype,args.trainiters) 
    env,test_opponents = get_env_and_test_opps(args.trun)
    opponents = get_all_opponent_params(opponent_pairs,test_opponents)

    prewashout_trials, postwashout_trials = [],[] #will save this data separately then concatenate
    for i in range(len(opponent_pairs)): #compare opponent pairs separately
        o = deepcopy(opponents[i]) #there was an overwriting bug here, deepcopy solves it
        prewashout_blocks, postwashout_blocks = [], []
        for j in range(args.ntests): #
            l,j = np.random.choice(len(o),size=(2)) #select two random opponents (to switch between)
            prewashout, postwashout = test_specific_opponents(env, model, [o[l],o[j]],nwashout = args.nwashout)
            #prewashout and postwashout are both lists of TrialTestObjects
            prewashout_blocks.extend(prewashout)
            postwashout_blocks.extend(postwashout)

        prewashout_trials.append(prewashout_blocks)
        postwashout_trials.append(postwashout_blocks)
    
    avg_switches,uppers,lowers = get_switching_behavior(prewashout_trials,postwashout_trials,opponent_pairs)
    plot_switching_behavior(opponent_pairs, avg_switches,uppers,lowers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and Test the Agent's Performance on how it switches from one opponent to another")

    parser.add_argument("--trun", type=int, required=False, default=86, help="model run # of the \
                                                                        trained multi-task agent")
    parser.add_argument("--mtype", required=False, default = "SSP", help="type of model (SSP or notSSP)")
    
    parser.add_argument("--trainiters", required=False, default="8000k", help="length of model training to be used (number+k)")
    parser.add_argument("--opponents", type=str, required=False, nargs = "+", default=["epsilonqlearn","patternbandit","1","softmaxqlearn"], 
                        help="pairs of opponents")
    parser.add_argument("--ntests", required = False, default = 50, type = int, help = "how many blocks to test opponent switch behavior") 
    parser.add_argument("--nwashout",type=int, required = False, default = 10, help = "number of washout blocks for testing")                

    args = parser.parse_args()
    opponent_pairs = [[args.opponents[i],args.opponents[i+1]] for i in range(0,len(args.opponents),2)] #split up opponents into ordered pairs

    test_opponents_switch(args,opponent_pairs)