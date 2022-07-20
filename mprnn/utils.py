import numpy as np
import pickle
from pathlib import Path
import gym
from neurogym import spaces
from mprnn.sspagent.a2c_custom import CustomA2C
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

from neurogym.wrappers import pass_reward
BETTER_NAMES = {"lrplayer":"Linear Combination","1":"MP 1st",'all':"MP 1st+2nd",
                'reversalbandit':"anti-correlated",'patternbandit':"Patterns",
            'softmaxqlearn':'Softmax QL','epsilonqlearn':'Epsilon-Greedy QL','mimicry':"n-back mimickry"}
FILEPATH = Path(__file__).parent.parent.absolute()

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

     
