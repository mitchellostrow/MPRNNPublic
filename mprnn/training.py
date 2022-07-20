
from pathlib import Path
from neurogym.wrappers import pass_reward
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import LstmPolicy
from stable_baselines import A2C
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from mprnn.sspagent.advanced_rnn import AuxLstmPolicy
from mprnn.utils import PassAction
from mprnn.sspagent.a2c_custom import CustomA2C
import gym

rng = np.random.default_rng()

def convert_dist_to_params(params):
    '''convert opponent_dist_kwargs into a dict of lists, which have the parameters to draw from
    the input will either be script_kwargs['env']['opponent_params] or overall_params
    Input:
        params (dict): from json, dictionary of environment params-- for ease of coding, these have structure of 'opponent':{'parameter':
                                                                                                {'start':x,'stop':y','step':z}}
    Output:
        params (dict): the same, but convert start,stop,step to a list mode.
    '''
    list_params = {}
    opp_params = params['env']['opponents_params']
    for opp,param_dict in opp_params.items():
        list_params[opp] = {}
        for name,st in param_dict.items():
            start = st['start']
            stop = st['stop']
            step = st.get('step',1)
            values = np.round_(np.arange(start,stop,step),2)
            list_params[opp][name] = values
    params['env']['opponents_params'] = list_params
    return params

def train_env_json(params,useparams = True):
    '''
    Using parameters, make an environment 
    Input: 
        params (dict): the paramters to make the environment from. This should be of structure train_params['env'] 
        useparams (boolean): whether or not to use the default opponent
    Output:
        env (neurogym environment)
    '''
    opponent_dist_kwargs = params['opponents_params']
    opp = rng.choice(params['train_opponents']) #randomly draw at the start
    if opp == "lrplayer":       
        #because this doesn't work with the same random draw, we need to specifically draw the parameters
        opp_kwargs = {}
        l_choice = rng.choice(opponent_dist_kwargs[opp]['len_choice'])
        l_outcome = rng.choice(opponent_dist_kwargs[opp]['len_outcome'])
        draw_random_params = lambda x,data: [rng.choice(data) for _ in range(x)]
        opp_kwargs['choice_betas'] = draw_random_params(l_choice,opponent_dist_kwargs[opp]['choice_betas'])
        opp_kwargs['outcome_betas'] = draw_random_params(l_outcome,opponent_dist_kwargs[opp]['outcome_betas'])
    else:
        opp_kwargs = {k:rng.choice(v) for k,v in opponent_dist_kwargs[opp].items()}
    #need to convert eh show_opp parameter to a boolean to input into the model    
    show_opp = bool(params['show_opp'] in {"true","True","TRUE"})
    if useparams:
        env = gym.make('mp-v0',show_opp=show_opp,episodic=True,
                        reset_time = params['reset_time'],opponent=opp,
                        opponents = params['train_opponents'],
                        opp_params = opponent_dist_kwargs,**opp_kwargs)
    else:
        env = gym.make('mp-v0',show_opp=show_opp,episodic=True,
                        reset_time = params['reset_time'],opponent=opp,
                        **opp_kwargs)
    env = pass_reward.PassReward(env)
    env = PassAction(env) #custom passaction that codes as one-hot representation
    env = DummyVecEnv([lambda: env])
    return env

def nn_json(env,params):
    '''
    Initializes an A2C agent (with SSP as CustomA2C) with the specified environment and parameters
    Input:
        env (neurogym environment)
        params (dict): of the format train_params, must have dictionary argument 'net_params']
    Returns:
        model (A2C agent or CustomA2C)
    '''
    nn_params = params['net_params']
    act_funs = {"tanh":tf.nn.tanh, "relu":tf.nn.relu}

    policy_kwargs = {
            'n_lstm': nn_params['nrec'], 
            'feature_extraction':"mlp", 
            "act_fun":act_funs[nn_params['act_fun']],
            'net_arch':nn_params['net_arch']
             }
    if not nn_params['SSP']:
        nn_params['net_args'].pop('pred_coef', None) #pred_coef is only useful for the SSP agent
        model = A2C(LstmPolicy, env, verbose=2, lr_schedule = 'constant', \
                    policy_kwargs = policy_kwargs,
                    max_grad_norm = 2, **nn_params['net_args']
                   )
    else:
        model = CustomA2C(AuxLstmPolicy, env, verbose=2, lr_schedule = 'constant', \
                    policy_kwargs = policy_kwargs,
                    max_grad_norm = 2,obs_pred_index = 0,**nn_params['net_args'] 
                    #obs_pred_index predicts the 0th index of the state vector (opponent choice)
                   )
    return model

def train_net(model,steps,name="",loc=None):
    '''
    Trains, saves, and returns model
    Inputs:
        model (stable-baselines): untrained
        steps (counting number): number of training steps
    Outputs:
        model: trained
    '''
    print("MODEL IS:", model,name,loc)
    model.learn(total_timesteps = steps,log_interval=100,tb_log_name="test_run")
    if loc is None:
        model.save(f"model{name}.zip")
    else:
        model.save(Path(loc,f"model{name}.zip"))
    return model

