
import numpy as np
import neurogym as ngym
from neurogym import spaces
from .matching_pennies import matching_pennies
from .misc_opponents import SoftmaxQlearn,EpsilonQlearn,ReversalBandit,PatternBandit,LRPlayer,MimicryPlayer

rng = np.random.default_rng()

class MPEnv(ngym.TrialEnv):

    def __init__(self, dt=100, timing=None, rewards = None, train = True,show_opp = True,
                opponent = "all",opponents=None,episodic=False,reset_time = 150,opp_params=None,**opp_kwargs):
        super().__init__(dt=dt)
        # Rewards
        self.rewards = {'abort' : -1, 'correct': +1., 'select':0.0, 'fail': 0.0}
        self.model_hist = [] #agent choice history
        self.reward_hist = [] #agent reward history
        self.seen = False #used for non-episodic setup
        self.opponent_action = None #saves the value of the opponent to compar against
        self.viewer = None #for visualization
        self.pright_mp = [] #for mp opponent
        self.biases = [] #for mp opponent
        self.opp_kwargs = opp_kwargs #sets opponent parameters for the particular opponent
        self.opp_params = opp_params #is the distribution of opponent parameters (lists), if none, refer to default
        self.opponent_name = str(opponent)
        self.train = train #whether or not to update agent
        self.updated = False #for non-episodic setup, in case agent fails to make it to outcome epoch
        self.show_opp = show_opp #include one-hot representation in state
        self.episodic = episodic
        self.testing_opp_params = None
        self.matching_pennies = matching_pennies
        self.all_opponents = ['lrplayer','1','patternbandit','all','reversalbandit','epsilonqlearn','softmaxqlearn','mimicry'] #list opponent names
        if opponents is None:
            self.opponents = self.all_opponents
        else:
            self.opponents = opponents
        self.probs = [i/len(self.opponents) for i in range(len(self.opponents)+1)]  #uniform cdf of opponents for random draw, for now
        self.base_reset_time = self.reset_time = reset_time #reset opponent every reset_time
        
        self.set_opponent(opponent,**opp_kwargs)
        self.opponent_ind = self.all_opponents.index(self.opponent_name)
        if rewards:
            self.rewards.update(rewards)
  
        if self.episodic:
            self.timing = {'outcome':dt}
            self.act_dict = {'left': 0, 'right': 1}
            self.action_space = spaces.Discrete(2,name = self.act_dict) 
            self.last_opp_choice = None
        else:
            self.timing = {
                    'fixation':100, 
                    'choice': 100,
                    'outcome':100}
            if timing:
                self.timing.update(timing)
            self.act_dict = {'fixation': 0, 'left': 1, 'right': 2}
            self.action_space = spaces.Discrete(3,name = self.act_dict)      
       
        #add in past_reward, past_action as observations
        if self.show_opp:
            if self.episodic:
                self.ob_dict = {'stimulus':[0,1],'opponent':list(range(2,2+8))}
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2+8,),
                                            dtype=np.float32,name = self.ob_dict)
            else:
                self.ob_dict = {'fixation':0,'stimulus':[1,2],'opponent':list(range(3,3+8))}
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3+8,),
                                            dtype=np.float32,name = self.ob_dict)
        else:
            if self.episodic:
                self.ob_dict = {'stimulus':[0,1]}
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32,name = self.ob_dict)
            else:
                self.ob_dict = {'fixation':0,'stimulus':[1,2]}
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32,name = self.ob_dict)
                 
    def __str__(self):
        return f"matching pennies with timing {self.timing}"

    def set_opponent(self,opponent,**opp_kwargs):
        '''
        Given an opponent and arguments, set the environment opponent and related parameters to that opponent
        '''
        if opponent == "softmaxqlearn":
            opp = SoftmaxQlearn(**opp_kwargs)
        elif opponent == "epsilonqlearn":
            opp = EpsilonQlearn(**opp_kwargs)
        elif opponent == "patternbandit":
            opp = PatternBandit(**opp_kwargs)
        elif opponent == "reversalbandit":
            opp_kwargs['rtrain'] = self.train #need to add in this variable to make sure updates work as expected here
            opp = ReversalBandit(**opp_kwargs)
        elif opponent in ['all',0,1,2,'0','1','2']:
            opp = opponent
        elif opponent == "lrplayer":
            opp = LRPlayer(**opp_kwargs)
        elif opponent == "mimicry":
            opp = MimicryPlayer(**opp_kwargs)
        else:
            raise ValueError("Agent type not found")
        self.opponent_name = str(opponent)
        self.opponent = opp
        self.opp_kwargs = opp_kwargs
        self.opponent_ind = self.all_opponents.index(str(opponent))

        return self.opponent_name,self.opponent,self.opp_kwargs,self.opponent_ind
    
    def set_testing_opponents(self,opp_params):
        '''
        save a list of (string,dict) tuples that set the opponent parameters that we wish to test on
        Then, when generating test runs, we can iterate over these on demand
        '''
        self.testing_opp_params = opp_params

    def draw_opponent(self, n = -1):
        '''
        Draws the opponent type and random parameters from a distribution
        '''
        if n >= 0 and self.testing_opp_params is not None:
            return self.set_opponent(self.testing_opp_params[n][0], **self.testing_opp_params[n][1])
        
        x = rng.random()
        for t,i in enumerate(self.probs):
            if x > i and x <= self.probs[t+1]:
                opponent = self.opponents[t]
                self.opponent_ind = self.all_opponents.index(str(opponent)) #was = t
                #this should keep the indices fixed regardless of what set of opponents are being used
                break
        opp_kwargs = {}
        gen = self.opp_params is not None  #if we are using the generated versions rather than the default
        if gen:
            params = self.opp_params[str(opponent)] #params is a dict of lists of the set of parameters to draw from
        if opponent in ["all",0,1,2,'0','1','2']:
            if gen:
                bias = params['bias'] #getting the dict of the range
                depth = params['depth']
                opp_kwargs['bias'] = 0 if bias == [] else rng.choice(bias) #random uniform draw according to params
                opp_kwargs['depth']= 4 if depth == [] else rng.choice(depth)
            else: #default range
                opp_kwargs['bias'] = rng.choice([0,-0.02,-0.05,0.02,0.05],p = [0.7,0.1,0.05,0.1,0.05])
                opp_kwargs['depth'] = rng.choice([2,4,8],p=[0.3,0.5,0.2])
            opp_kwargs['p'] = 0.05
        if opponent == 'patternbandit':
             #get rid of the binary numbers with only 1s ,*range(16,31)
            if gen:
                lengths = params['length']
                length = rng.choice(lengths) #random draw of the length (this makes it evenly distributed)
                l = list(range(2**(length-1),2**length-1)) #ex: length 5 = range(16,31) --> 2**4: 2**5 - 1
            else:
                l = [*range(4,7),*range(8,15)]
            opp_kwargs['pattern'] = bin(rng.choice(l))[2:]

        if opponent == "reversalbandit":
            if gen:
                pr = params['pr']
                update = params['update']
                opp_kwargs['pr'] = rng.choice(pr)
                opp_kwargs['update'] = rng.choice(update)
            else:
                opp_kwargs['pr'] = rng.choice(np.arange(0.1,1,0.1))
                opp_kwargs['update'] = rng.choice([50,75,100,125])
            opp_kwargs['rtrain'] = self.train
        if type(opponent) == str and opponent[-6:] == "qlearn":
            if gen:
                lr = params['lr']
                gamma = params['gamma']
                bias = params['bias']
                opp_kwargs['lr'] = rng.choice(lr)
                opp_kwargs['gamma'] = rng.choice(gamma)
                opp_kwargs['bias'] = rng.choice(bias)
            else:
                opp_kwargs['lr'] = rng.choice([0.25,0.5,1])
                opp_kwargs['gamma'] = rng.choice([0.5,0.6,0.75,0.9,0.99])
                opp_kwargs['bias'] = rng.choice([0,-0.02,-0.05,0.02,0.05],p = [0.5,0.1,0.15,0.1,0.15])
        if opponent == "epsilonqlearn":
            if gen:
                eps = params['epsilon']
                opp_kwargs['epsilon'] = rng.choice(eps)
            else:
                opp_kwargs['epsilon'] = rng.choice([0.0,0.1,0.2,0.5],p =[0.33,0.23,0.22,0.22])
        if opponent == "softmaxqlearn":
            if gen:
                temp = params['temp']
                opp_kwargs['temp'] = rng.choice(temp)
            else:
                opp_kwargs['temp'] = rng.choice([0.1,0.5,1,2,3])
        if opponent == "lrplayer":
            if gen:
                b = params['b']
                len_choice = params['len_choice']
                len_outcome = params['len_outcome']
                choice_betas = params['choice_betas']
                outcome_betas = params['outcome_betas']
                opp_kwargs['b'] = rng.choice(b)
                l_choice = rng.choice(len_choice)
                l_outcome = rng.choice(len_outcome)
                draw_random_params = lambda x,data: [rng.choice(data) for _ in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(l_choice,choice_betas)
                opp_kwargs['outcome_betas'] = draw_random_params(l_outcome,outcome_betas)
            else:
                opp_kwargs['b'] = rng.choice([0,0,0,0,0,0,0,0,0,-0.1,0.1,-0.25,0.25,0.5,-0.5])
                draw_random_params = lambda x: [round(rng.choice([0,0,0,0,0,0,-1,1,-0.7,0.7,-0.5,0.5,
                                                                        -0.4,0.4,-1.5,1.5,-2,2]),1) for i in range(x)]
                opp_kwargs['choice_betas'] = draw_random_params(rng.choice(np.arange(3)))
                opp_kwargs['outcome_betas'] = draw_random_params(rng.choice(np.arange(3)))
        if opponent == "mimicry":
            if gen:
                n = params['n']
                opp_kwargs['n'] = rng.choice(n)
            else:
                opp_kwargs['n'] = rng.choice(np.arange(0,5))

        return self.set_opponent(opponent,**opp_kwargs)

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        Here you have to set:
        The trial periods: fixation, stimulus...
        Optionally, you can set:
        The ground truth: the correct answer for the created trial.
        """
        self.done = False
        self.action = None
        if len(self.model_hist) > self.reset_time and self.train:
            #if we are ready to reset
            avgchoice = sum(self.model_hist)/len(self.model_hist)
            avgrew = sum(self.reward_hist)/len(self.reward_hist)
            #print an update
            #print(f"opponent {self.opponent}, avg choice {avgchoice} avg reward {avgrew} on {len(self.reward_hist)} trials")
            self.clear_data()
            #draw new opponent
            self.draw_opponent()
            self.reset_time = self.base_reset_time + rng.random() * self.base_reset_time //2 #change the reset time according to a random walk with mean reset_time
            if self.testing_opp_params is not None:
                self.reset_time = self.base_reset_time
            self.done = True
        
        if self.opponent_name in ['all',0,1,2,'0','1','2']: #matching pennies is just a function, so we can call it here
            bias = self.opp_kwargs.get('bias',0)
            depth = self.opp_kwargs.get('depth',4)
            pval = self.opp_kwargs.get('p',0.05)
            opponent_action,pchooseright,biasinfo = matching_pennies(self.model_hist,self.reward_hist,depth,pval,self.opponent_name,bias) 
        else: #other opponents are class based, so we use the step function to update the model with the most recent choices and get the output
            if self.updated: #take a step if we've made it there
                opponent_action,pchooseright,biasinfo = self.opponent.step(self.model_hist[-1],self.reward_hist[-1]) #should consider just calculating over history
            else: #otherwise, get the last values saved as they won't have changed
                opponent_action,pchooseright,biasinfo = self.opponent.get_last_saved()

            if self.opp_kwargs != self.opponent.opp_kwargs:
                self.opp_kwargs = self.opponent.opp_kwargs #fix the saved opponent kwargs if the agent has changed parameters

        self.updated = False
        self.pright_mp.append(pchooseright)
        self.pr = pchooseright
        self.biases.append(biasinfo)   
        self.seen = False
        trial = {'opponent_action': opponent_action}
        #convert {0,1} action to one hot vector to add into observation 
        stim_action = [0,0]
        if self.episodic:
            if self.last_opp_choice is not None:
                stim_action[self.last_opp_choice] += 1
            self.last_opp_choice = opponent_action
        else:
            stim_action[opponent_action] += 1
        self.opponent_action = opponent_action

        if self.episodic:
            #only use the outcome period as the single step in the episode
            self.add_period(['outcome'])
            self.set_groundtruth(opponent_action,'outcome')
        else:
            self.add_period(['fixation', 'choice','outcome']) 
            self.add_ob(1,'fixation',where = 'fixation')  
            self.set_groundtruth(0,'fixation')
            self.set_groundtruth(opponent_action+1,'choice')
            self.set_groundtruth(opponent_action+1,'outcome')

        if self.show_opp:
            opps = [0]*8
            opps[self.opponent_ind] = 1 #one-hot representation of opponent
            self.add_ob(opps,'outcome',where = 'opponent')
        self.add_ob(stim_action,'outcome', where = 'stimulus')
    
        return trial

    def clear_data(self): 
        '''effectively reset the environment
        '''
        self.model_hist = []
        self.reward_hist = []
        self.pright_mp = []
        self.biases = []
        self.updated = False

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        obs = self.ob_now
        reward = 0
        done = False
        if not self.episodic:
            if self.in_period('fixation') and action != 0:
                reward = self.rewards['abort'] #abort reward is -1, but don't actually abort here
            if self.in_period('choice'):
                if not self.seen and action != 0:
                    #need to map from (1,2) to (0,1)
                    self.action = action - 1
                    self.seen = True
                elif not action or (self.seen and action-1 != self.action):
                    reward = self.rewards['abort']
                    new_trial = True
                if not new_trial:
                    reward = self.rewards['select']
        if self.episodic:
            self.action = action 
            self.seen = True
            self.updated = True

        if self.in_period('outcome'):
            #we will always be here for episodic
            pr = self.pr
            #reversal_bandit is a 2-bandit task rather than a matching-pennies task
            #this adds the probabilistic condition that the agent can win without the "correct choice" 
            rev_bandit_condition = (self.opponent == 'reversalbandit' and rng.random() < (self.action*(pr) + (1-self.action)*(1-pr)))
            if (not self.seen or action-1 != self.action) and not self.episodic:
                reward = self.rewards['abort']
                flag = False
            elif self.action == self.opponent_action or rev_bandit_condition:
                reward = self.rewards['correct']
                self.performance = 1
                flag = True
            else:
                reward = self.rewards['fail']
                flag = True
            if flag:
                self.reward_hist.append(reward)
                self.model_hist.append(self.action)
                done = self.done
            new_trial = True
        info = {'new_trial': new_trial}# 'gt': self.opponent_action}
        #changed 3rd parameter to new_trial in order to see if mean episode length occurs
        return obs, reward, done, info
   
