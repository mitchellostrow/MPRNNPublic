import numpy as np

#this file contains classes of opponents to be played against in the matching pennies game
#they must implement the step function
rng = np.random.default_rng()
class Opponent():
    def __init__(self,**kwargs):
        self.opp_kwargs = kwargs
        self.opponent_action = rng.choice([0,1])
        self.pchooseright = 0.5
        self.biasinfo = None
        self.bias = kwargs.get('bias',0)
        self.act_hist = [self.opponent_action]
    def get_last_saved(self):
        return self.opponent_action, self.pchooseright,self.biasinfo
    def step(self,choice,rew):
        '''
        choice is the agent choice (the deep RL model) and its reward
        '''
        raise NotImplementedError

class Qlearn(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.lr = kwargs.get('lr',0.1)
        self.Qs = np.zeros((2))
        self.max_choices = []
        self.gamma = kwargs.get('gamma',0.99)

    def step(self,choice,rew):
        rew = 1 - rew #if opponent wins, agent loses--rew is agent reward, so flip bit
        self.RPEupdate(rew)
        opponent_action, pchooseright, biasinfo = self.action()
        self.opponent_action = opponent_action
        self.pchooseright = pchooseright
        self.biasinfo = biasinfo
        self.act_hist.append(opponent_action)
        return opponent_action,pchooseright,biasinfo

    def action(self):
        raise NotImplementedError

    def RPEupdate(self, reward):
        """Reward Prediction Error Calculation and Update of Q values"""
        #rescorla-wagner removes the maxQ(s',a) term in RPE
        #self.state is S (past state), (self.act,self.reward) is S' (present state)
        Qsa = self.Qs[self.act_hist[-1]] #q-value of the action just taken
        maxQ_Sprime = max(self.Qs)
        self.Qs[self.act_hist[-1]] = Qsa + self.lr * (reward + self.gamma * maxQ_Sprime - Qsa)
        
class SoftmaxQlearn(Qlearn): #softmax exploration q learning agent
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.temp = kwargs.get('temp',1)

    def __str__(self):
        return 'softmaxqlearn'

    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
       
        self.ql, self.qr = self.Qs
        self.max_choices.append(np.argmax(self.Qs))
        #softmax to find probability
        self.pr = 1 / (1 + np.exp(- self.temp * (self.qr - self.ql))) #softmax probability
        self.pr += self.bias 
        
        if not self.pr:
            self.pr = 10 ** -10
        action = 1 if rng.random() < self.pr else 0 #random choice

        return action, self.pr, action_data

class EpsilonQlearn(Qlearn): #epsilon-greedy q learning agent
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.eps = kwargs.get('epsilon',0.1)

    def __str__(self):
        return 'epsilonqlearn'

    def action(self):
        action_data = {'Qs':self.Qs,'bias':self.bias}
       
        self.ql, self.qr = self.Qs
        self.max_choices.append(np.argmax(self.Qs))
        self.pr = 1 / (1 + np.exp(-(self.qr - self.ql)))

        if rng.random() < self.eps: #epsilon-greedy 
            action = rng.choice([0,1]) #1 - self.max_choices[-1]
        else:
            action = self.max_choices[-1]

        return action, self.pr, action_data

class ReversalBandit(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.base_updates = self.updates = kwargs.get('updates',30)
        self.train = kwargs.get('rtrain',False)
        self.episode_count = 0
        self.pr = kwargs.get('pr',0.9)
        self.max_choices = []
        self.ps = []

    def step(self,choice,rew):
      
        self.episode_count += 1
        self.max_choices.append(1 if self.pr > 0.5 else 0) #what would be the best choice to make?
        self.ps.append(self.pr)
        action = 1 if rng.random() < self.pr else 0 #actual choice of the opponent
        self.act_hist.append(action)
        self.opponent_action = action
        self.pchooseright = self.pr
        self.biasinfo = {'updates':self.updates,'ep_count':self.episode_count,'bias':self.bias}
        return action, self.pr, self.biasinfo

class PatternBandit(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.pattern = kwargs.get('pattern','01') #fixed pattern to play
        self.plen = len(self.pattern)
        self.count = 0
        self.max_choices = []

    def __str__(self):
        return f'patternbandit with pattern {self.pattern}'

    def step(self,choice,rew):
        ind = self.count % self.plen #where in the pattern are we?
        action = int(self.pattern[ind])
        self.max_choices.append(action)
        self.count += 1 #step forward
        self.act_hist.append(action)
        self.opponent_action = action
        self.pchooseright = 1 if action else 0
        self.biasinfo = None
        return action, self.pchooseright,self.biasinfo

class LRPlayer(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.deterministic = kwargs.get("deterministic",False)
        self.b = kwargs.get('b',0)
        self.choice_betas = kwargs.get('choice_betas',[0]) # left to right is earliest to latest
        self.outcome_betas = kwargs.get('outcome_betas',[0]) #ex: [...,t-3,t-2,t-1,t]
        self.lc, self.lo = len(self.choice_betas),len(self.outcome_betas)
        #if we don't care about one of the parameters, just fill it with a zero so LR still works
        if self.lo > 0:
            self.rews_inframe = [0] * self.lo #stores the last n rewards for the regression: (left = -1, right = 1)
        else:
            self.rews_inframe = [0]
            self.outcome_betas = [0]

        if self.lc > 0:
            self.acts_inframe = [2*self.opponent_action-1] + [0] * (self.lc-1)
        else:
            self.acts_inframe = [0]
            self.choice_betas = [0]

    def __str__(self):
        return f'Logistic Regression Player with bias {self.b}, choice betas {self.choice_betas}, outcome betas {self.outcome_betas}'

    def step(self,choice,rew):
        rew = 2 * rew - 1 #convert reward to {-1,1}
        self.rews_inframe.pop(0) #pop off the front of the q (this is technically O(n) but fixed n)
        self.rews_inframe.append(rew*(2*self.act_hist[-1]-1)) #outcome is reward*action or opponent choice

        betas = np.hstack([self.choice_betas,self.outcome_betas]).T #(1 x n)
        inputs = np.hstack([self.acts_inframe,self.rews_inframe]) #stack the input values n-back into a vector (n x 1)
        out = betas @ inputs + self.b #linearly apply parameters
        self.pchooseright = 1 / (1 + np.exp(-out)) #sigmoid function to get probability
        if self.deterministic:
            action = 1 if self.pchooseright >= 0.5 else 0
        else:
            action = 1 if rng.random() < self.pchooseright else 0 #take action
        self.acts_inframe.pop(0)
        self.acts_inframe.append(2*action-1) #same thing as the other operation
        #it's a queue, so the most recent elem is at the end
        self.act_hist.append(action)

        return action,self.pchooseright,None

class MimicryPlayer(Opponent):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.n = kwargs.get("n",1)
        self.acts_inframe = [] #we don't care about act_hist, but rather only the last n choices
    def __str__(self):
        return f"t-{self.n} mimicry"
    
    def step(self,choice,rew):
        self.acts_inframe.append(choice) 
        if len(self.acts_inframe) >= self.n: #if we have played more than n trials, play the opposite choice that the agent took
            return 1 - self.acts_inframe.pop(0),self.pchooseright,None
        else:
            return rng.choice([0,1]),self.pchooseright,None

        

