import itertools
import os
from pathlib import Path
import pickle as pkl
import numpy as np
from mprnn.sspagent.a2c_custom import CustomA2C
from mprnn.utils import setup_run, get_env_and_test_opps,get_net,iterate_over_opponents, FILEPATH

def feed_seq(model,seq,_states,dones):
    '''
    feeds in the given sequence to the model and saves the output policy and state
    '''
    for o in range(0,len(seq),2): #every two is a single state
        act,rew = int(seq[o]),int(seq[o+1])
        correct_choice = act ^ rew #xor is the solution to this
        #format of the state is one-hot real answer, reward, onehot action
        obs = [[1-correct_choice,correct_choice,rew,1-act,act]]
        #feed the deterministic state into the agent
        if o == len(seq)-2: 
            #if we're on the last state,get action probability
            pi, _ = model.proba_step(obs,_states,dones)[0][1], model.value(obs,_states,dones)[0]
        _, _states = model.predict(obs,state = _states,deterministic = False, mask = dones)

    return pi, _states

def feed_seq_and_save_state(env,model,opponent,seq,nblocks=1):
    '''
    it plays the model against the opponent for ~75 iterations 
    (variable number centered on 75)
    (till good convergence of opponent)
    then inputs the fixed sequence of states to the network
    extracts the network recurrent state and its policy output on the final state
    saves these and averages this over a number of runs
    should also have the option not to average in case we want to look at populations
    Parameters:
        env,model (obvious)
        opponent (tuple: (str,dict)): detailing opponent name and parameters
        seq (str): detailing the state or sequence of inputs we give to the network
                        variable length. I.e. "1001" is 2 states, read left to right
                        in order of time, first is rewarded choose left, second is
                        unrewarded choose right.
        nblocks (int): number of runs to test and average states 
    Returns:
        avg_pi (float): average policy output for a given input sequence
        avg_state (np.array): average agent state after the given input sequence
    '''
    if opponent[0] == "reversalbandit":
        opponent[1]['update'] = 150
    env.envs[0].set_testing_opponents([opponent]) #i think this needs to be a list
    env.envs[0].train = False
    avg_pi = 0
    avg_state = np.zeros(model.initial_state.shape[-1])
    for _ in range(nblocks):
        obs = env.reset()
        env.envs[0].clear_data()
        env.envs[0].draw_opponent(0)
        model.set_env(env) 
        dones,_states,_ = setup_run(model)

        for _ in range(np.random.randint(50,80)):
            action, _states = model.predict(obs,state = _states,deterministic = False, mask = dones)
            if isinstance(model,CustomA2C):
                #auxiliary prediction agent gives a tuple of output (action,[pred])
                action,pred = action
                pred = pred.squeeze()
            obs, _, dones, _ = env.step(action)
        pi, _states = feed_seq(model,seq,_states,dones)
        #save _states and pi to the vector
        avg_state += 1/nblocks * _states.squeeze() #states is 1,256
        avg_pi += 1/nblocks * pi
    return avg_pi, avg_state

def get_repvecs(env,model,opponent,nblocks,seqlength):
    '''
    Given a particular length of sequences, runs the feed_seq_and_save_state function for each sequence
    and saves the output policy and state into a list for each sequence
    Returns:
        policy_rep, state_rep (np.arrays): outputs of the representation for each sequence
    '''
    seqs = ["".join(seq) for seq in itertools.product("01", repeat=seqlength*2)]
    #this gives all the different input state sequences of length 5
    policy_rep = np.zeros(len(seqs))
    state_rep = np.zeros((len(seqs),model.initial_state.shape[-1])) #TBD: use the whole state or just cell/hidden?
    for i,seq in enumerate(seqs):
        avg_pi,avg_state = feed_seq_and_save_state(env,model,opponent,seq,nblocks=nblocks)
        policy_rep[i] = avg_pi
        state_rep[i] = avg_state

    return policy_rep, state_rep

def save_repvecs_across_opponents(args):
    '''
    finally, put all the pieces together: 
    iterate over the opponents, iterate over sequences, iterate multiple times
    Arguments:
        args (from command line)
            runindex,modeltype,trainiters: as before, specifies model and opponents
            seqlength (int): how many trials to use to define sequences of states
            nblocks (int): number of runs to average each representation over
            npop (int): number of representations to add into each population. 
                These will be compared among one another
    Returns:   
        policy_reps (list): list of numpy arrays for each opponent. 
            each numpy array has shape (npop,seqlength) 
    '''
    model,_ = get_net(args.runindex,args.modeltype,args.trainiters)  
    env,test_opponents = get_env_and_test_opps(args.runindex)
    opponent_order = [] #save the order in which we see opponents
    #that can then be used to map to the other policy_reps and state_reps
    #using this instead of a dictionary because we can use matrix operations on a numpy array 
    numopps = 0
    for _ in iterate_over_opponents(test_opponents):
        numopps += 1 #count the number of opponents
    seqlength = 2*args.seqlength
    policy_reps = np.zeros((numopps,args.npop,2**seqlength))
    state_reps = np.zeros((numopps,args.npop,2**seqlength,model.initial_state.shape[-1]))
    for i,opp in enumerate(iterate_over_opponents(test_opponents)):
        print("testing opponent ",opp)
        for j in range(args.npop):
            p,s = get_repvecs(env,model,opp,args.nblocks,args.seqlength) 
            policy_reps[i,j] = p.squeeze()#2**seqlength x 1
            state_reps[i,j] = s.squeeze()#2**seqlength x 128
        opponent_order.append(opp)
    opponent_order = np.array(opponent_order)
    return policy_reps.squeeze(),state_reps.squeeze(),opponent_order

def save_reps_to_pkl(policy_reps,state_reps,opponent_order,args):
    try:
        os.mkdir(Path(FILEPATH,"data","RSAtrials"))
    except FileExistsError:
        pass
    with open(Path(FILEPATH,"data","RSAtrials",f'model{args.runindex}{args.modeltype}{args.trainiters}_npop{args.npop}.pkl'), 'wb') as f: 
        pkl.dump([policy_reps,state_reps,opponent_order], f) 

def load_reps(name):
    '''
    name is the dataset pickle name str that we're reading from
    '''
    with open(Path(FILEPATH,"data","RSAtrials",f"{name}.pkl"), "rb") as input_file: 
        representation = pkl.load(input_file) 
    print("loaded representations")
    policy_reps,state_reps,opponent_order = representation
    state_reps = np.array([i.squeeze() for i in state_reps])
    return policy_reps,state_reps,opponent_order