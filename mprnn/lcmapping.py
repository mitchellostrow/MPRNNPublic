import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from mprnn.utils import get_env_and_test_opps
from mprnn.testing import test_specific_opponents
from mprnn.representation import get_repvecs

def get_design_mat(trialdata,length=3):
    '''
    Given the saved trial data (from test_specific_opponent), makes a design matrix
    of our data X to fit the logistic regression to
    Arguments:
        trialdata (list(TrialTestObject)): data saved from the runs
        length (int): number of past time points to include in the regression, or
                the length of the window we look for fitting our model
    Returns:
        design_mat (np.array): input data for the LR, actions at time t-1,t-2,...t-length
        y (np.array): output data for the LR, action the agent took at time t
    '''
    design_mat = []
    y = []
    for block in trialdata:
        opp_rew_b = 2*block.rewards-1 #block[1] is the sequence of agent rewards, so map to {-1,1}
        opp_choice_b = 1 - 2*(np.array(np.logical_xor(block.rewards,block.actions),dtype=np.float64))
        l = len(opp_rew_b)
        for i in range(length,l-1): #have to cut off the first length trials
            observation = []
         
            rew_window = opp_rew_b[i-length:i] #flipped to have the same structure as the lrplayer
            choice_window = opp_choice_b[i-length:i]
            rew_choice_interaction = rew_window * choice_window
            observation.extend(choice_window)
            observation.extend(rew_choice_interaction)
            
            design_mat.append(observation)
            y.append(opp_choice_b[i]/2 + 0.5)

    design_mat = np.array(design_mat)
    y = np.array(y)
    return design_mat,y

def fit_lr(design_mat,y,C):
    '''
    Given the,x,y and a regularization term, fits the model
    Returns the training accuracy and the parameters (we're not using the model again)
    '''
    clf = LR(penalty="l1",solver="liblinear",C=C,max_iter=400).fit(design_mat,y)
    return clf.score(design_mat,y),clf.intercept_,clf.coef_

def fit_best_lr(trialdata):
    '''
    Tries different regularization terms for the logisitc regression, saves the best one
    Hopefully identifies sparse parameters
    Returns:
        params (tuple(int,float)):  hyperparameters that best fit the model
        coefs (tuple(float,np.array)): linear parameters of he best fit LC mapping model
        maxscore (float): training accuracy of the LR model
    '''
    maxscore = 0.0
    params = None
    coefs = None
    for l in range(1,6):
        #most likely, more LC fits will be < score which will mean more things will extrapolate rather than interpolate
        design_mat,y = get_design_mat(trialdata,l)
        for c in [0.01,*np.arange(0.01,1.0,0.1)]: #test a range of different hyperparameters
            try:
                score,intercept,coef = fit_lr(design_mat,y,c)
            except ValueError: #if all y values are part of 1 class (all 1's or zeros), then fit just the intercept
                coef = [[0]*2*l]
                pr = np.mean(y)
                intercept = [10 * (-1+2*pr)]
                #if pr = 0, then intercept = -10 (sigmoid(-10) = 0), otherwise pr = 1 and intercept = 10
                score = 1.0
            if score>maxscore:
                maxscore = score
                params = (l,c)
                coefs = (intercept,coef)
    return params,coefs,maxscore

def get_lc_mapping(model,opponent,args,ntrials=2):
    '''
    Given a particular opponent, fit the best lc model using fit_best_lr
    Arguments:
        model (A2C/CustomA2C): agent
        opponent (tuple(str,dict)): opponent name and parameters to fit
        args (ArgumentParser): from command line
        ntrials (int): how much data to save
    Returns:
        policy_rep (np.array): representations of the policy for each sequence
        state_rep (np.array): hidden state rep for each sequence
        maxscore (float): best training accuracy (this is the lcmap score used for filtering)
        opp_approx (tuple("lrplayer",dict)): best fit lc opponent params
    '''
    env,_ = get_env_and_test_opps(args.runindex)

    if opponent[0] == "reversalbandit": #we don't need to fit data to this, we can just define the intercept
        pr = opponent[1]['pr']
        if pr == 1.0: #b would be infinity, so just set high and the sigmoid fxn will do the rest (0.999)
            b = 1000
        elif pr == 0.0:
            b = -1000
        else:
            b = np.log(pr/(1-pr))
        opp_approx = ("lrplayer",{'b':b,
                              'choice_betas':[0.0],
                              'outcome_betas':[0.0]})
        maxscore = 1.0 #if we modelled it, the score would actually be pr,but we know this is an optimal mapping
    elif opponent[0] == "lrplayer":
        opp_approx = opponent #and lC agent's best fit is itself
        maxscore = 1.0
    else:
        trialdata = [] 
        for _ in range(ntrials):
            prewashout, postwashout = test_specific_opponents(env, model, [opponent],nwashout = 1 )
            trialdata.extend(prewashout)
            trialdata.extend(postwashout)
        params,coefs,maxscore = fit_best_lr(trialdata)
        l = params[0]
        opp_approx = ("lrplayer",{'b':coefs[0][0],
                                  'choice_betas':list(np.round_(coefs[1][0][:l],decimals=3)),
                                  'outcome_betas':list(np.round_(coefs[1][0][l:],decimals=3))})
        
    #don't forget to reverse the coefficients to be in the right order [t-0,t-1,...t-n]!
    policy_rep,state_rep = get_repvecs(env,model,opp_approx,args.nblocks,args.seqlength)  #don't forget to multiply by 2 here
    #default numbers for seqlength (2*seqlength), and nruns similar to those in the datawe're comparing
    return policy_rep,state_rep.squeeze(),maxscore,opp_approx    