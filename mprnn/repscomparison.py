import numpy as np
from pathlib import Path
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from mprnn.representation import load_reps
from mprnn.lcmapping import get_lc_mapping
from mprnn.utils import get_net, FILEPATH

class RepsComparison():
    def __init__(self,args):
        '''
        This class will be used to calculate and store the distances between representation vectors
        It works for both the LC mapping-ood distance and the wd-ood distance
         if args.usecenterdists, then calculates the distances between average sequence representation
            otherwise, calculates the average distance between identical sequence representations for different opponents
        '''
        self.args = args
        self.get_model()
        self.get_ood_wd_repvecs()
        self.get_ood_wd_distances()
        self.get_avgandmindist_ood()
        self.separate_reps()

    def get_model(self):
        '''
        Loads the model and saves it for future runs of get_avgandmindist_ood
        '''
        self.model,_ = get_net(self.args.runindex,self.args.modeltype,self.args.trainiters)
    
    def get_ood_wd_repvecs(self):
        '''
        Loads the reps with load_reps then identifies the test opponents and separates them
        '''
        _, self.state_reps, self.opponent_order = load_reps(self.args.savedrepspath) 
        rep = self.state_reps 
        if len(rep.shape) == 4 and self.args.usecenterdists:
            #i.e., the reps are saved by population, not averaged already
            #dimension is number of opponents, npop, number sequences, hidden dim
            rep = rep.mean(axis=1) 

        self.avged_state_reps = np.array(rep)
        self.avged_state_reps = self.avged_state_reps.mean(axis=1) #average over the sequences (that's what I had to do previously)
        #if recurrent, shape is 503 (number of opponents), 64 (number of sequences), 256 (hidden state dimension)
        #if we're using center dists, average over sequences, so it will be 503,256
        self.get_ood_wd_inds()
        self.ood_reps,self.wd_reps = self.avged_state_reps[self.ood_inds], self.avged_state_reps[self.wd_inds]

    def get_ood_wd_inds(self):
        '''
        identifies the indices in opponent order that correspond to ood and wd opponents
        Right now the type of ood opponents is static, because we only tested one type of model
        '''
        within = []
        ood = []
        opponent_names = []
        ood_opponents = {'lrplayer','1','patternbandit'}
        for i,(opp,_) in enumerate(self.opponent_order):
            if opp in ood_opponents:
                within.append(i)
            else:
                ood.append(i)
            opponent_names.append(opp)
        self.opponent_names = np.array(opponent_names)
        self.wd_inds = np.array(within)
        self.ood_inds = np.array(ood)
        
    def get_ood_wd_distances(self,dist="euclidean"):
        '''
        Calculates the distances between all ood and wd opponents. used for min dists
        '''
        if dist == "euclidean":
            distfunc = euclidean_distances #pairwise
        if self.args.usecenterdists:
            self.dists = distfunc(self.ood_reps,self.wd_reps) 
            #these are the distances between the centers, or average representation of the 64 sequences of fixed input
            #ood opponents is the first dimension, wd opponents is the second dimension
            #now get the average distance between individual sequences of each rep
        else:
            self.get_avg_dist_bw_seqs()
    
    def get_avg_dist_bw_seqs(self):
        '''
        If we're not comparing the center of the representations for all sequences, compare each sequence
        against one another and average. This is different than the distance between the average because
        the euclidean distance is a nonlinear function in x and y
        '''
        self.dists = np.zeros((self.ood_inds.shape[0],self.wd_inds.shape[0])) #distance matrix between each ood and wd opponent
        for i,x in enumerate(self.ood_inds):
            d1 = self.state_reps[x]
            seqlength = len(d1)
            for j,y in enumerate(self.wd_inds):
                d2 = self.state_reps[y]
                distbw = np.zeros((seqlength)) #distance between each sequence for the two opponents
                for k in range(seqlength):
                    d1r,d2r = d1[k],d2[k]
                    distbw[k] = euclidean(d1r,d2r)
                self.dists[i,j] = distbw.mean()
        
    def get_opposite_dists(self,opponent):
        '''
        given an opponent, gets the distances from that opponent to the other class, all opponents
        Arguments:
            opponent (tuple(str,dict)): opponent name and parameters
        Returns:
            dists (1-d np.array): distances of that opponent to the other class
            class_inds (1-d np.array): indices of that opponent class
        '''
        index = np.where(self.opponent_order==opponent)[0] #index of the opponent in the overall list of opponents
        inood = np.sum(self.ood_inds == index) #1 if in the ood_indices list
        class_inds = [self.wd_inds,self.ood_inds]
        if inood:
            dists = self.dists
        else:
            dists = self.dists.T #dists is (ood,wd)
            #transpose so we can use the same axis for either case later on
        #get the index in the distance vector that corresponds to the opponent index to extract the distance vector
        #for that opponent
        distindex = np.argmax(class_inds[inood] == index) 
        #class_inds[inood] is class_inds[1] = self.ood_inds if inood is True
        dists = dists[distindex]
        return dists,class_inds[1-inood] #get the opposite index here for get_mindist to get the right index
    
    def get_mindist(self,opponent):
        '''
        Given a particular opponent, gets the minimum distance between its representation and the other class
        Arguments:
            opponent (tuple(str,dict)): opponent name and parameters in dictionary
        Returns:
            mindist (float): distance
            minopponent (tuple(str,dict)): name and parameters of the minimizing opponent
        '''
        #first identifies if opponent is in wd or ood
        dists,class_inds = self.get_opposite_dists(opponent)   
        minopp,mindist = np.argmin(dists),np.min(dists) #index and value of the opposing class opponent that is minimum
        minopp_ind = class_inds[minopp] #find the self.opponent_order index of the minimum opponent from the opposing indices
        minopponent = self.opponent_order[minopp_ind]
        
        return mindist,minopponent
    
    def avg_dist_all(self,opponent):
        '''
        if opponent is in the within dist class, gets the average distance to ood class, and vice versa
        '''
        return np.mean(self.get_opposite_dists(opponent)[0])
    
    def get_lc_mapping_distance(self,index,opponent):
        '''
        Given an opponent, fits an LC model to it and identifies the distance between that opponent and the lc model representation
        '''
        _,state_rep,maxscore,opp_approx = get_lc_mapping(self.model,opponent,self.args)
        if self.args.usecenterdists:
            lc_rep = state_rep.mean(axis=0)
            normal_rep = self.avged_state_reps[index] #average state rep
            dist = euclidean(lc_rep,normal_rep)
        else:
            normal_rep = self.state_reps[index]
            seqlength = len(normal_rep)
            distbw = np.zeros((seqlength))
            for k in range(seqlength):
                d1r,d2r = normal_rep[k],state_rep[k] 
                #iterate over each index and calculate the euclidean distance bw them
                distbw[k] = euclidean(d1r,d2r)
            dist = distbw.mean() 
        return dist,opp_approx,maxscore
    
    def get_avgandmindist_ood(self,calculate_lcmap=True,opponent=None):
        '''
        computes the min distance opponent, name, and average opponent distance of all ood opponents to the wd opponents
        Arguments:
            calculate_lcmap (bool): whether or not to calculate the LC mapping (time intensive)
            opponent (tuple(str,dict)): if not None, calculate the avg and min dist only for this opponent       
        '''
        if opponent is None:
            self.mindists = np.zeros((self.ood_inds.shape[0]))
            self.avgdists = np.zeros((self.ood_inds.shape[0]))
            if calculate_lcmap:
                self.LCmap_dists = np.zeros((self.ood_inds.shape[0]))
                self.LCmap_opponents = []
                self.LCmap_scores = np.zeros((self.ood_inds.shape[0]))

            self.minopponents = []
            self.ood_opponents = []

        n = 0
        for i,opp in enumerate(self.opponent_order):
            if i not in self.ood_inds:
                continue
            if opponent is not None and opp != opponent:
                continue
            print(i,opp)
            mindist,minopponent_name = self.get_mindist(opp)
            self.minopponents.append(minopponent_name)
            self.mindists[n] = mindist
            self.avgdists[n] = self.avg_dist_all(opp)
            self.ood_opponents.append(opp)
            if calculate_lcmap:
                dist,opp_approx,score = self.get_lc_mapping_distance(i,opp)
                self.LCmap_dists[n] = dist
                self.LCmap_opponents.append(opp_approx)
                self.LCmap_scores[n] = score
                
            n += 1 

    def separate_reps(self):
        '''
        Given the distances, lc mapping score, and cutoff distances and scores, separates
        ood opponents out into template match, interpolate, and extrapolate, saves in self.index_dict
        also saves within-dist in the dict
        '''
        self.index_dict = {}
        tm = self.mindists < self.args.cutoffdist
        interp = np.logical_and(np.logical_and(self.LCmap_dists < self.args.cutoffdist,self.LCmap_scores > self.args.cutoffscore),
                                                self.mindists > self.args.cutoffdist)
        extrap = np.logical_and(np.logical_or(self.LCmap_dists > self.args.cutoffdist,self.LCmap_scores < self.args.cutoffscore),
                                    self.mindists > self.args.cutoffdist)

        self.index_dict['templatematch'] = self.ood_inds[tm]
        self.index_dict['interpolate'] = self.ood_inds[interp]
        self.index_dict['extrapolate'] = self.ood_inds[extrap]
        self.index_dict['within-dist'] = self.wd_inds

    def save(self):
        self.model = None #can't save the model in the numpy binary so we need to remove it (could use pickle here)
        try:
            np.save(Path(FILEPATH,"data","RSAtrials",f"repscomparisonobject_model{self.args.runindex}{self.args.modeltype}{self.args.trainiters}.npy"),self)
        except FileExistsError:
            pass
        self.get_model()
 
def load_repscomparison(args):
    if args.savedrepscomparisonpath is not None:
        reps = np.load(Path(FILEPATH,"data","RSAtrials",f"repscomparisonobject_{args.savedrepscomparisonpath}.npy"),allow_pickle=True).reshape(1)[0]
        reps.args = args
        reps.separate_reps()
        reps.get_model()
    else:
        assert args.savedrepspath is not None

        reps = RepsComparison(args)
    return reps