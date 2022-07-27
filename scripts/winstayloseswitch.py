import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.ndimage import gaussian_filter

from mprnn.testing import test_specific_opponents
from mprnn.utils import get_net, set_plotting_params, get_env_and_test_opps

set_plotting_params()

def test_WSLS(args):
    '''
    Given input arguments to the scrip, test the agent on win-stay/lose-switch variations 
    saves data, then analyze and plot
    Arguments:
        args (ArgParse): arguments from script:
            trun: model run index of the trained agent
            mtype: whether the model is SSP or default A2C
            trainiters: how long the model has traned for
            ntests: how long the model should be tested for
    Returns:
        None
    '''
    model,_ = get_net(args.trun,args.mtype,args.trainiters) 
    env,_ = get_env_and_test_opps(args.trun)

    b2s = np.arange(-5,5,0.5)
    stayprob = np.zeros((args.ntests*2,len(b2s)))
    rewprob = np.zeros((args.ntests*2,len(b2s)))
    for i in range(args.ntests):
        for j,b2 in enumerate(b2s):
            opponent = [("lrplayer",{'b': 0.0, 'choice_betas': [0.0, 0.0], 'outcome_betas': [b2, 0.0]})]
            prewashout, postwashout = test_specific_opponents(env, model, opponent,nwashout = 1)
            #recall that the tuple data structure is acts,rews,states,preds,opponents,opp_classes,opp_kwargs,obs
            for k,data in enumerate([prewashout,postwashout]):
                data = data[0] 
                stay = data.actions[:-1]==data.actions[1:] #stay
                assert len(data.actions) == len(data.rewards)
                pstay = np.sum(stay) /len(data.actions)
                rew = np.sum(data.rewards)/len(data.rewards) #win
                index = i + args.ntests*k
                stayprob[index,j] = pstay
                rewprob[index,j] = rew

    #plot
    truepwsls = 1 / (1 + np.exp(b2s)) 
    plt.rcParams['font.size'] = 10

    _, axes = plt.subplots(1,1,figsize=(7,5),sharey=True)

    for j,data in enumerate([stayprob,rewprob]):
        if j == 0:
            l = "Stay"
        elif j == 1:
            l = "Reward "
        m = gaussian_filter(data.mean(axis=0),sigma=1)
        stderr = np.std(data,axis=0) / len(data) #stderr
        upper = m + 0.5*stderr
        lower = m - 0.5*stderr

        axes.plot(truepwsls,m,label=l)
        axes.plot(truepwsls,upper, color='tab:blue', alpha=0.1)
        axes.plot(truepwsls,lower, color='tab:blue', alpha=0.1)
        axes.fill_between(truepwsls, lower, upper, alpha=0.2)

    plt.ylim(0.1,1.0)
    axes.set_xlabel("$P_{opp}$(Win-Stay/Lose-Switch)")
    axes.set_ylabel("Probability")
    axes.axline((0.0, 1.0), slope=-1,c="black",label="Optimal")

    axes.legend(fontsize=10)#loc="upper left")
    plt.tight_layout()
    plt.savefig(Path("results","WSLSbehavior.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and Test the Agent's Performance on how it switches from one opponent to another")

    parser.add_argument("--trun", type=int, required=False, default=86, help="model run # of the \
                                                                        trained multi-task agent")
    parser.add_argument("--mtype", required=False, default = "SSP", help="type of model (SSP or notSSP)")
    
    parser.add_argument("--trainiters", required=False, default="8000k", help="length of model training to be used (number+k)")

    parser.add_argument("--ntests", required = False, default = 50, type = int, help = "how many blocks to test opponent switch behavior") 

    args = parser.parse_args()
    test_WSLS(args)