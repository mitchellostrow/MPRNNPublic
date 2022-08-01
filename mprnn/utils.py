import numpy as np
import argparse
import pickle
from pathlib import Path
import gym
import json
import matplotlib.pyplot as plt
import tensorflow as tf

from neurogym import spaces
from neurogym.wrappers import pass_reward
from sklearn.model_selection import ParameterGrid
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import LstmPolicy
from stable_baselines import A2C

from mprnn.sspagent.advanced_rnn import AuxLstmPolicy
from mprnn.sspagent.a2c_custom import CustomA2C

tf.compat.v1.disable_eager_execution()

rng = np.random.default_rng()

BETTER_NAMES = {"lrplayer":"Linear Combination","1":"MP 1st",'all':"MP 1st+2nd",
                'reversalbandit':"anti-correlated",'patternbandit':"Patterns",
            'softmaxqlearn':'Softmax QL','epsilonqlearn':'$\epsilon$-Greedy QL','mimicry':"n-back mimickry"}
FILEPATH = Path(__file__).parent.parent.absolute()

def default_argparser(details):
    '''
    arguments that need to be parsed for each script. Add on other details after
    Arguments:
        details (str): description of the argparsers
    Returns:
        parser (ArgumentParser): the parser
    '''
    parser = argparse.ArgumentParser(description=details)

     
    parser.add_argument("--runindex", type=int, required=False, default=86, help="model run # of the \
                                                                        trained multi-task agent")
    parser.add_argument("--modeltype", required=False, default = "SSP", help="type of model (SSP or notSSP)")
    
    parser.add_argument("--trainiters", required=False, default="8000k", help="length of model training to be used (number+k)")
    parser.add_argument("--nblocks",type=int, required=False, default=50, help="number of trial blocks to test each opponent on")
    
    
    parser.add_argument("--nwashout",type=int,required=False,default = 30,help="length of washout blocks")
    #linear classifier arguments (for perturb and lineardecoding)
    parser.add_argument("--clf",required=False,default= "logistic", help = "'logistic','SVC'): logistic regression or support vector classifier")
    parser.add_argument("--penalty",default="l2",required = False, help = "regularization term for linear classifier")
    parser.add_argument("--train_split",default=0.7,type=float,help = "percent of prewashout_trials to use as training data")
    return parser

def set_plotting_params(fontsize = 15):
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rc('xtick', labelsize='medium')
    plt.rc('ytick', labelsize='medium')
    plt.rc('font', family='serif')

def convert_names_short(opps):
    '''
    Convert long names that are used internally to pretty looking names
    Arguments:
        opps (list or single string):opponent names that are in the values of BETTER_NAMES
    Returns:
        the same data structure but with the converted names
    '''
    if type(opps) in {list,np.array,np.ndarray}:
        assert [i in BETTER_NAMES.keys() for i in opps]
        return np.array([BETTER_NAMES[i] for i in opps])
    elif type(opps) == str:
        return BETTER_NAMES[opps]

def read_pickle(loc,data=None):
    '''
    Reads some data from a pickle file
    Arguments:
        loc (Path): location of the file in the directory
        data (optional): data that we may wish to append onto
    Returns:
        data (list): list data read from the .p file
    '''
    if data is None:
        data = []
    with (open(loc, "rb")) as openfile:
        while True:
            try:
                data.append(pickle.load(openfile))
            except EOFError:
                break
    return data

def get_net(runindex,name,time):
    '''
    Given a particular network index, type, and training time, loads and returns the model
    Arguments:
        runindex (int,str): index code of the model, or a directory path
        name (str): model type: SSP or notSSP
        time (int or str): training time in 1000s of episodes
    Returns:
        model (A2C or CustomA2C): model
        loc (Path): directory of the model
    '''
    if type(time) == int:
        time = str(time) + "k"
    if type(runindex) != int and "\\" in runindex: #if we're going into a new directory list it
        loc = Path(runindex,name,time)
    else:
        loc = Path(f"run{runindex}",name,time)
    modelpath = Path(FILEPATH,"data","models",loc,Path(f"model{time}.zip"))
    if name == "SSP":
        model = CustomA2C.load(modelpath) 
    else:
        model = A2C.load(modelpath) 
    return model,loc

def setup_run(model,_states=None):
    '''
    Given the model, return some relevant features to start saving the data
    Returns:
        dones (list of bools): whether or not the block has ended
        _states (np.array): recurrent state
        nrec (int): size of the recurrent state
    '''
    #set up model run 
    dones = [False]
    if _states is None:
        _states = model.initial_state
    nrec = len(_states.reshape(-1))
    return dones,_states,nrec

def get_env(**envdata):
    '''
    Loads environment according to a dictionary rather than a json, just for testing
    '''
    opp = envdata.get('opp','1')
    opp_kwargs = envdata.get('opp_kwargs',{'bias':0,'depth':4,'p':0.05})
    timing = envdata.get('outcome',100)
    episodic = envdata.get('episodic',True)
    show_opp = envdata.get('show_opp', True)
    train = envdata.get('train',False)
    opponents = envdata.get('opponents',None)
    opp_params = envdata.get('opp_params',None)
    reset_time = envdata.get('reset_time',150) #this shouldn't matter if we're just testing

    env = gym.make('mp-v0',timing = timing,opponent = opp,train = train,episodic = episodic,
                show_opp = show_opp,opponents = opponents, opp_params=opp_params,reset_time = reset_time,**opp_kwargs)
    env = pass_reward.PassReward(env)
    env = PassAction(env)
    env = DummyVecEnv([lambda: env])

    return env

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

def get_env_from_json(params,useparams = True):
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

def get_env_and_test_opps(runindex):
    '''
    As for most tests before, we need to load our environment, and testing opponents
    runindex (int or path (str with "\\" )): indicating which run number to load our agent from
    '''
    if type(runindex) != int and "\\" in runindex: #if we're going into a new directory, list it
        loc = Path(runindex)
    else:
        loc = Path(f"run{runindex}")

    with open(Path(FILEPATH,"data","models",loc,Path("train_params.json") ) ) as f:
        script_kwargs = json.load(f)
    train_params = convert_dist_to_params(script_kwargs)
    test_opponents = []
    #other_opponents = [[]]
    for k,v in train_params['env']['opponents_params'].items(): #add train opponents first
        #if k in {'lrplayer','1','patternbandit'}:
         #   test_opponents.append( (k,v) )
        #else:
         #   other_opponents.append( (k,v) )
        test_opponents.append((k,v))
    #test_opponents.extend(other_opponents)
    env = get_env_from_json(train_params['env'],useparams=False)
    return env,test_opponents

def iterate_over_opponents(test_opponents):
    '''
    Given the list of tuples (str,dict) detailing opponents and various parameters,
    produces a generator (yield) that iterates over all opponent + parameter combos
    Arguments:
        test_opponents ((str,dict)): details the name of the opponent in the string and the parameters to iterate over in the dict
    Yields:
        (curropp,params): tuple of a particular opponent and it's parameters
    '''
    for opp in test_opponents:
        curropp = opp[0]
        params = opp[1]
        #need to modify this for lrplayer because we need to select coefficients
        #also need to modify this for pattern because we need to select the specific pattern
        #for lrplayer we need to generate the list of parameters over each combination of lengths
            #it would be fastest to go with the largest length of everything and then just iterate over that
            #because 0's in the later positions are equivalent to shorter distance
        if curropp == "patternbandit":
            smallest = min(params['length'])
            largest = max(params['length'])
            l = list(range(2**(smallest-1),2**largest-1)) 
            l = [bin(i)[2:] for i in l if i not in [7,16,31,63]]
            params = {'pattern':l}
        elif curropp == "lrplayer":
            #need to sample each possible beta coefficient for each time delay
            lchoice = max(params['len_choice']) 
            loutcome = max(params['len_outcome'])
            choiceparams = {f'c{i}':params['choice_betas'] for i in range(lchoice)}
            outcomeparams = {f'o{i}':params['outcome_betas'] for i in range(loutcome)}
            total_coefs = {'choice_betas':[],'outcome_betas':[]}
            for coefs in ParameterGrid(choiceparams):
                total_coefs['choice_betas'].append([coefs[f'c{i}'] for i in range(lchoice)])
            for coefs in ParameterGrid(outcomeparams):
                total_coefs['outcome_betas'].append([coefs[f'o{i}'] for i in range(loutcome)])
            total_coefs['b'] = params['b']
            params = total_coefs

        param_grid = ParameterGrid(params)
        for params in param_grid:
            #given these parameters
            yield (curropp,params) 

class PassAction(gym.Wrapper):
    """Modifies observation by adding the previous action."""

    def __init__(self, env):
        self.env = env
        env_oss = env.observation_space.shape[0]
        self.num_act = env.action_space.n
        try:
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(env_oss+self.num_act,),
                                                dtype=np.float32,name = self.ob_dict)
        except AttributeError: #if no ob_dict
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(env_oss+self.num_act,),
                                                dtype=np.float32)
        try:
            self.action_space = spaces.Discrete(self.num_act,name = self.act_dict)
        except AttributeError:
            self.action_space = spaces.Discrete(self.num_act)

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        actions = np.zeros(self.num_act)
        actions[action] = 1
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, actions))
        return obs, reward, done, info

class TrialTestObject():
    '''
    Stores data saved from one trial of the model in an interpretable way
    Why not a pandas dataframe? not all of our data is numeric, also this builds off of legacy code
    Essentially this stores the data in a more interpretable way.
    Attributes:
        actions: a list or list of lists of actions that the agent took (in sequences {a_0,a_1,..a_n})
        rewards: a list or list of lists corresponding to those actions
        states: a list of numpy arrays or ... corresponding to the resulting recurrent states from those actions
        predicted_opponent_actions: for the self-supervised prediction agent, this is a list or list of lists of predicted opponent actions
        opponent_kwargs: a list of dictionaries containing the hyperparameters of the different agents
        opponent_names: a list of strings containing the names of each opponent the agent played. Same length as actions
        opponent_indices: a list of ints containing the index code of each opponent played
        observations: a list of vectors or list of lists of vectors containing the output of the environment after each action.
    '''
    def __init__(self,actions=None,rewards=None,states=None,preds=None,opps_kwargs=None,opps=None,opp_inds=None,obs=None):
        init = lambda d: [] if d is None else d
        self.actions = init(actions)
        self.rewards = init(rewards)
        self.states = init(states)
        self.predicted_opponent_actions = init(preds)
        self.opponent_kwargs = init(opps_kwargs)
        self.opponent_names = init(opps)
        self.opponent_indices = init(opp_inds)
        self.observations = init(obs)

    def __getitem__(self,ind):
        ''''
        For indexing, returns a new object of the same type so we can continue to use the structure.
        '''
        return TrialTestObject(self.actions[ind],self.rewards[ind],self.states[ind],
                            self.predicted_opponent_actions[ind],self.opponent_kwargs[ind],
                            self.opponent_names[ind],self.opponent_indices[ind])

     
