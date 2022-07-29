from pathlib import Path
import os
import json 
import random
from copy import copy

import numpy as np
import tensorflow as tf
from mprnn.sspagent.a2c_custom import CustomA2C

from mprnn.utils import TrialTestObject, FILEPATH
from mprnn.utils import get_net, get_env, setup_run,get_env_from_json,convert_dist_to_params
tf.compat.v1.disable_eager_execution()

def get_ood_opponents(params):
    '''
    Calculate the opponent opponents that are out of distribution via set complements
    Inputs:
        params (dict): of the form train_params (has to have ['env']['train_opponents])
    Outputs:
        list of strategies that are not in params
    '''
    all_strats = set(["lrplayer","1","all","reversalbandit","patternbandit", #set of all strategies, train_opponents \in all_strats
                        "epsilonqlearn","softmaxqlearn","mimicry"])
    used_strats = set(params['env']['train_opponents'])                    
    return list(all_strats.symmetric_difference(used_strats)) #this is (A intersection B)^C  

def get_ood_params(train_params,overall_params):
    '''
    This function compares the overall range of parameters to the train range of parameters 
    And outputs the out of distribution portion of parameters to be used in generalization testing
    Inputs:
        train_params (dict): of type train_params.json, but must be post-processing in convert_dist_to_params
        overall_params (dict): of type overall_params.json, but also must be post processing
    Output:
        ood_params (dict): of type train_params['env']['opponents_params']
    '''
    train_params = train_params['env']['opponents_params']
    overall_params = overall_params['env']['opponents_params']
    ood_params = {}
    for opp,params in overall_params.items(): #loop through each opponent
        ood_params[opp] = {}
        for param, vals in params.items():
            tp_set = set(train_params[opp][param])
            overall_set = set(vals)
            ood_params[opp][param] = list(overall_set.symmetric_difference(tp_set))
    return ood_params

def step_and_save(env,model,**envdata):
    '''
    Takes a single step of the model and the environment, and outputs all relevant features
    Arguments:
        env (MPEnv): game environment
        model (A2C/CustomA2C): agent
        envdata (dict): contains features for iterating the environment and the model (observation,model state, done boolean)
    Returns:
        data (tuple): contains the action,model prediction,new observation,agent reward, done boolean, recurrent state
    '''
    pred = None
    action, _states = model.predict(envdata['obs'],state = envdata['_states']
                                ,deterministic = False, mask = envdata['dones'])
    if isinstance(model,CustomA2C):
        #auxiliary prediction agent gives a tuple of output (action,[pred])
        action,pred = action
        pred = pred.squeeze()
    obs, reward, dones, _ = env.step(action)
    data = (action,pred,obs,reward,dones,_states)
    return data

def make_perturbation(recurrent_state,perturb):
    '''
    Given a state of the model, moves it in a particular direction
    Arguments:
        recurrent_state (np.array): vector representation of the LSTM state
        perturb (dict): 
            coefs (np.array): vector of coefficients for how the bases correspond to the recurrent space
            params (np.array): new expected coefficients on the bases
            vector (np.array): if we wish to reset to a particular vector
            reset (bool): if we are fully resetting to vector
    Returns:
        recurrent_state (np.array): perturbed state
    '''
    s = recurrent_state.shape[1]//2
    if perturb['reset']:
        #to make a basis, can do QR decomposition, but then moving them to the
        #right spot on the coefficients 
        #Q,R = np.linalg.qr(perturb['coefs'].T) #Q is the orthonormal basis
        decoding_coefs = perturb['decoding_coefs']
        #define projection matrix as 
        c = [np.dot(decoding_coefs[i],recurrent_state[0,:s]) for i in range(decoding_coefs.shape[0])]
        #perturb['coefs'] @ recurrent_state[0,:s].T  #3x128 * 128 x 1
        #only the first half is the relevant part, this is now 3 x 1, the coefficients of the state on the basis
        print("original state is", c)
        for _ in range(40):
            p = -sum([c[i] * decoding_coefs[i] for i in range(decoding_coefs.shape[0])])
            #1x3 * 3x128, reduced rank version in full sapce
            #I don't want to have to loop this but it doesn't work well otherwise--the accuracy doesn't have to be perfect
            #subtract the different axes of the vector of the current state in this subspace
            recurrent_state[0,:s] += p
            c = [np.dot(decoding_coefs[i],recurrent_state[0,:s]) for i in range(decoding_coefs.shape[0])]
            #print(c, "should be 0,0,0")
            if np.sum(np.square(c)) <= 0.01:
                break
        recurrent_state[0,:s] += perturb['vector']
        c = np.array([np.dot(decoding_coefs[i],recurrent_state[0,:s]) for i in range(decoding_coefs.shape[0])])
        print("new state is ", c, "should be", perturb['new_coefs_to_perturb_to'])
    else:
        recurrent_state[0,:s] += perturb['vector']
    return recurrent_state

def iter_step_and_save(env,model,perturb = None,**startdata):
    '''
    The inner loop. Runs a model for a single block of length startdata{'steps'}
    Arguments:
        env (MPEnv): test environment
        model (A2C/CustomA2C): model
        perturb (optional, dict): in case we want to perturb the state of the agent, contains instructions on how
        startdata (dict): features on which to start saving data from the run
    Returns:
        trialdata (TrialTestObject): recoreded data from a single block
    '''
    nrec = startdata['nrec']
    obs = startdata['obs']
    recurrent_state = startdata['_states']
    dones = startdata.get('dones',[False])
    steps = startdata.get('steps',150)
    acts,rews,states,preds = np.zeros(steps),np.zeros(steps),np.zeros((steps,nrec)),np.zeros(steps)
    opponents,opp_classes,opp_kwargs = [],np.zeros(steps),[]
    saved_obs = np.zeros((steps,1,obs.shape[-1])) #save all observations, the 1 is in there to make sure it's vectorized
    saved_obs[0] = obs
    states[0] = recurrent_state.reshape(-1)
    for _ in range(steps):
        envdata = {'obs':obs,'_states':recurrent_state,'dones':dones}
        data = step_and_save(env,model,**envdata)

        action,pred,obs,reward,dones,recurrent_state = data

        if perturb and perturb['time_of_perturbation'] == _: 
            #add perturbation at correct time and opponent
            recurrent_state = make_perturbation(recurrent_state,perturb)
            
        if pred is not None:
            preds[_] = pred
        acts[_] = action[0]
        rews[_] = reward
        opponents.append(env.envs[0].opponent_name)
        opp_classes[_] = env.envs[0].opponent_ind
        opp_kwargs.append(env.envs[0].opp_kwargs)
        if _ < steps-1: 
            saved_obs[_+1] = obs 
            states[_+1] = recurrent_state.reshape(-1)

    opponents = np.array(opponents)
    opp_kwargs = np.array(opp_kwargs)
    trialdata = TrialTestObject(acts,rews,states,preds,opps=opponents,
                            opps_kwargs=opp_kwargs,opp_inds=opp_classes,obs=saved_obs)
    return trialdata

def get_model_states(model,**envdata):
    '''
    Given a particular set of values, runs the agent niters time for steps steps and returns the average results
    How is this better than generate_data? This allows us to generate a test environment with a particular set of parameters and rerun to average
    Inputs:
        model (stable-baselines agent)
        envdata(dict):
            opp (str): name of opponent to start (or play on)
            opp_kwargs (dict): dictionary of opponent arguments
            timing (dict): dictionary of timing, default is for episodic
            episodic (boolean): environment parameter specifiying episodic or temporal mode (with fixation, choice,delay,outcome periods)
            show_opp (boolean): does the environment include a one-hot representation of the opponent in its state?
            steps (int): number of steps, as in generate data
            train (boolean): set environment into train mode (opponent resets every 150 steps) or test mode (fixed agent)
            opponents (list): list of strings of opponents to be played in this environment -- None defaults to default opponents\\parameters
            opp_params (dict of dicts): various parameters (in list format) to be chosen for those opponents
    Outputs:
        see generate_data (except they are averages here across runs)
    '''
    steps = envdata.get('steps',150)
    env = get_env(**envdata)
    model.set_env(env) 
    obs = env.reset()
    env.envs[0].clear_data()
    dones,_states,nrec = setup_run(model)
    startdata = {'obs':obs,'dones':dones,'_states':_states,'nrec':nrec,'steps':steps}

    trialdata = iter_step_and_save(env,model,None, **startdata)
    return trialdata

def loop_get_model_states(env,model,params,nblocks):
    '''
    this function draws a random opponent from the given environment, then fixes that opponent and parameters
    in a new environment and test sthat for 150 steps with get_model_states. Running it this way to be certain
    About what opponent is playing so as to not mix up labels
    Arguments:
        env: environment with the test parameters
        model: stablebaselines A2C or CustomA2C agent
        params (dict): training parameters with keys;
            opponents_params (dict): opponent parameters
            train_opponents (list): opponents to train (or test) against
            reset_time: time for environment to reset (won't need in testing)
            show_opp (bool): whether or not to show the opponent one-hot encoding in the state representation
    Returns:
        blockdata (TrialTestObject): data structure with test data
    '''
    opps,opp_inds,opps_kwargs = [],[],[]
    states,actions,rewards,preds = [],[],[],[]
    for _ in range(nblocks):
        opp,_,opp_kwargs,opponent_ind = env.envs[0].draw_opponent() #get my new parameters
        opps.append(opp)
        opp_inds.append(opponent_ind)
        opps_kwargs.append(opp_kwargs)
        #adding these two values to the parameters tells the environment what opponent to start on,
        #and because get_model_states runs environments in test mode (no switching)
        #this is the only opponent played against
        params['opp'] = opp
        params['opp_kwargs'] = opp_kwargs
        trialdata = get_model_states(model,**params)

        states.append(trialdata.states)
        actions.append(trialdata.actions)
        rewards.append(trialdata.rewards)
        preds.append(trialdata.predicted_opponent_actions)

    repeat_data = lambda x, steps : np.array([i for i in x for _ in range(steps)])
    opp_inds = repeat_data(opp_inds,150) #repeating each value 150 times in a row to make all outputs the same length
    opps_kwargs = repeat_data(opps_kwargs,150)
    opps = repeat_data(opps,150)

    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)
    states = np.concatenate(states)
    blockdata = TrialTestObject(actions,rewards,states,preds,opp_kwargs,opps,opp_inds)
    return blockdata

def test_net(model,env,test_opponents=None,opponent_params=None,
        reset_time=150,steps_per_block=150,show_opp=False,nblocks=200):
    '''
    Tests the network on either within or out of distribution data
    Arguments:
        model (A2C/CustomA2C): agent
        env (MPEnv): environment
        environment_params (dict): some parameters to govern the environment behavior
        test_opponents (list): a list of opponents to test on
        opponent_params (list/dict): parameters for those opponents
        reset_time (int): how long to run each opponent for before resetting (should be > steps_per_block but not required)
        steps_per_block (int): how long to test the opponent for each block
        show_opp (bool): whether to include a one-hot rep of the opponent in the observation
        nblocks (int) how many blocks to test on.
    Returns:
        blockdata (TrialTestObject): data from the run
    '''
    if test_opponents is None:
        test_opponents = env.envs[0].opponents
        opponent_params = env.envs[0].opp_params
    params = {"opponents_params":opponent_params, "train_opponents":test_opponents,
                        "reset_time":reset_time,"show_opp":show_opp,'steps':steps_per_block}
    env = get_env_from_json(params) #reinitiate an environment with the given parameters and opponents
    blockdata = loop_get_model_states(env,model,params,nblocks)
    return blockdata

def get_gather_data(runindex,name):
    '''
    Given a particular opponent, loads the training/test data (details on the opponents), and the environment
    Arguments:
        runindex (int,str): index code of the model, or a directory path
        name (str): model type: SSP or notSSP
    Returns:
        training_times (list): available training times we can draw from
        train_params (dict): parameters of the environment, model, and of the training opponents
        ood_opponents (list): opponents that the agent was not trained on
        env (MPEnv): matching pennies environment
    '''
    assert name in {"SSP", "notSSP"}
    if type(runindex) != int and "\\" in runindex:
        #in case our file is in another directory, can load by specifying runindex as a directory
        loc = runindex
    else:
        loc = f"run{runindex}"
    datapath = Path(FILEPATH,"data")
    modelpath = Path(datapath,"models",loc)
    training_times = os.listdir(Path(modelpath,"SSP") )
    overall_params = json.load(open(Path(datapath,"params",Path("overall_params_dist.json") ) ) )
    with open(Path(modelpath,"train_params.json") ) as f:
        script_kwargs = json.load(f)

    train_params = convert_dist_to_params(script_kwargs)
    overall_params = convert_dist_to_params(overall_params)
    ood_opponents = get_ood_opponents(train_params)
    ood_params = get_ood_params(train_params,overall_params)

    env = get_env_from_json(train_params['env'],useparams=True) 

    return training_times,train_params,ood_opponents,ood_params,env

def gather_data_time(runindex,name,time,nblocks=800):
    '''
    Tests a particular network, indexed by runindex,name,and time, on both trained and untrained for a particular time
    Arguments:
        runindex (int,str): index code of the model, or a directory path
        name (str): model type: SSP or notSSP
        time (iint or str): training time in 1000s of episodes
    Returns:
        blockdata_within (TrialTestObject): block data for tests against the within-training-distribution opponents
        blockdata_ood (TrialTestObject): likewise, but for ood opponents
    '''
    _,train_params,ood_opponents,_,env = get_gather_data(runindex,name)
    model,_ = get_net(runindex,name,time)

    print(f"testing {name} {time}")
    blockdata_within = test_net(model,env,reset_time=env.envs[0].reset_time,nblocks=nblocks)
    blockdata_ood = test_net(model,env,ood_opponents,train_params['env']['opponents_params'],reset_time=env.envs[0].reset_time,nblocks=nblocks)
    return blockdata_within,blockdata_ood

def test_specific_opponents(env, model, opponents,nwashout = 30,perturb=None):
    '''
    Tests model on specific block test: 
    opponents[0],opponents[1],opponents[2], large set of random opponents, opponents[0],opponents[1],opponents[2]
    opponents is a list of tuples of (string,dict) detailing opponent name and params
    Returns the testing data from the first set of opponents and the last, after the washout set
    perturb (dict): details when, and how to perturb the state 
            parameters: 'time_of_perturbation': int-- which step in the block to make the perturbation
                        'opponent_index_to_perturb_from': int-- which opponent to perturb (0,1,2)
                        'vector': np.array -- the perturbation vector (adds to the state)
                        'reset': bool --  True if we want to reset activity on this axis to the provided value, false if we just want to add it
    '''
    ops = copy(opponents)
    env.envs[0].set_testing_opponents(opponents)
    obs = env.reset()
    env.envs[0].train = False
    model.set_env(env) 
    dones,_states,nrec = setup_run(model)
    
    startdata = {'obs':obs,'dones':dones,'_states':_states,'nrec':nrec,'steps':150}
    opponents_kwargs = [i[1] for i in opponents]
    def washout(nwashout):
        i = 0
        while i < nwashout:
            trialdata = iter_step_and_save(env,model,None,**startdata)#this runs one block so we don't need to redraw
            #step_tuple has structure (acts,rews,states,preds,opponents,opp_classes,opp_kwargs,obs)
            #if the opponnent is one of the specific test ones, skip
            if trialdata.opponent_kwargs[0] in opponents_kwargs:
                continue #using a while loop bc of this guy
            startdata['_states'] = trialdata.states[-1].reshape(1,trialdata.states[-1].shape[-1]) 
            startdata['obs'] = trialdata.observations[-1]
            i += 1
  
    washout(1)
    #first set of testing run
    random.shuffle(opponents)
    if perturb: #if it's not None or not False
        #this ensures that the index in perturb_dict.json actually means the same thing each run
        #and isn't swapped around from the random shuffling above
        i = opponents.index(ops[perturb['opponent_index_to_perturb_from']]) #where is the to-be-perturbed opponent in the list of opponents
        op = opponents[perturb['opponent_index_to_perturb_from']] #save the opponent that's in that position already
        opponents[perturb['opponent_index_to_perturb_from']] = opponents[i] #swap the two
        opponents[i] = op

    def test_opponents(opponents):
        trialobjects = []
        for i, _ in enumerate(opponents):
            env.envs[0].clear_data()
            env.envs[0].draw_opponent(i) #reset opponent and environment data
            if perturb and perturb['opponent_index_to_perturb_from'] == i: 
                trialdata = iter_step_and_save(env,model,perturb,**startdata)
            else:
                trialdata = iter_step_and_save(env,model,None,**startdata)
            #if first_state is not None and all(first_state == step_tuple[2][0]):
               # raise ValueError("not continuous across blocks")
            #acts,rews,states,preds,opponents,opp_classes,opp_kwargs,obs
            startdata['_states'] = trialdata.states[-1].reshape(1,trialdata.states[-1].shape[-1]) 
            startdata['obs'] = trialdata.observations[-1]
            #reset the state to the last one
            trialobjects.append(trialdata)
        return trialobjects

    prewashout_trials = test_opponents(opponents)
    #washout trials
    washout(nwashout)
    postwashout_trials = test_opponents(opponents) #retest after washout
    
    return prewashout_trials, postwashout_trials


