import numpy as np
from mprnn.baselines import get_all_baselines
from mprnn.utils import convert_names_short
from collections import OrderedDict

def make_avg_reward(blockdata,step=150):
    '''Averages the reward across episodes by each opponent. 
    Arguments:
        blockdata (TrialTestObject): saving data from the opponent. The following features we care most about:
            rewards (np.array): numiter * 150, episodic reward of agent
            opponent_names (list): numiter * 150, opponent names
            opponent_indices (list): numiter * 150, indices for classification
    Returns:
        opponent_rewards (OrderedDict): keys are the names of opponents, values are the average rewards for that opponent
    '''
    opponents = convert_names_short(blockdata.opponent_names) #make the names more l
    opponent_rewards = OrderedDict()
    for i in sorted(np.unique(blockdata.opponent_indices)): #currently 8 opponents, indexing by 0
        #identify the opponent with the index and separate out all the data
        rew_i = blockdata.rewards[blockdata.opponent_indices==i]
        opponent_i = opponents[blockdata.opponent_indices==i][0] 
        #get the mean rew for each episode (rew[i:i+step] is each episode) by separating out into a 2-d array and averaging over the row
        if len(rew_i) > step:
            rew_sep_by_play = np.array([rew_i[j:j+step] for j in range(0,len(rew_i)-step,step)]).mean(axis = 1)
        else:
            rew_sep_by_play = []
        opponent_rewards[opponent_i] = rew_sep_by_play
    return opponent_rewards

def make_dev(opponent_rewards,baseline=None):
    '''
    Given average reward by opponent (output of make_avg_reward), compares that against the mean baseline reward
    Arguments:
        opponent_rewards (OrderedDict): dictionary with opponent name: average reward per episode in data of test data
        baseline (TrialTestObject): saving data from some baseline runs (will also be passed through make_avg_reward)
    Returns:
        deviations_by_opp (OrderedDict): dictionary with opponent name: difference between average reward in test and baseline data
    '''
    deviations_by_opp = OrderedDict()
    if baseline is not None: 
        baseline_rewards = make_avg_reward(baseline)
        baseline_mus = OrderedDict({k:np.mean(v) for k,v in baseline_rewards.items()})
    else:
        baseline_mus = get_all_baselines()

    for opp in opponent_rewards.keys():
        if baseline_mus[opp] is None:
            deviations_by_opp[opp] = [-1]*10 #need something else here, not this
        else:
            deviations = np.array([rews - baseline_mus[opp] for rews in opponent_rewards[opp]])
            deviations_by_opp[opp] = deviations
    return deviations_by_opp

def make_dev_from_baseline(blockdata,baseline=None):
    '''
    given a TrialTestObject dataframe, especially array of awards and the opponents + opponent arguments, algorithms,
     calculate the average deviation from baseline.
    Arguments:
        blockdata (TrialTestObject): saved data from a test agent run
        baseline (optional, TrialTestObject): saved data from a baseline run
    Returns:
        deviations_by_opp (OrderedDict): dictionary separating the deviation between test and baseline reward by opponent
    '''
    opponent_rewards = make_avg_reward(blockdata)
    return make_dev(opponent_rewards,baseline)

