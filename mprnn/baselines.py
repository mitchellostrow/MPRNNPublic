from pathlib import Path
from mprnn.utils import FILEPATH, read_pickle

BASELINESPATH = Path(FILEPATH,"data","models","baselines")
def get_all_baselines(untrained=False):
    '''
    Extracts the saved dictionary of trained or untrained baselines into a dictionary {opponent_name:meanreward}
    Arguments:
        extractuntrained (bool): if True, will look for the untrained baseline distribution of mean reward
    Returns:
        data (dict, None): dictionary of opponent name: mean reward
    '''
    name = "baseline_dists"
    if untrained:
        name += "_untrained"
    data = None
    try:
        data = read_pickle(Path(BASELINESPATH,name+".p"))[0] #has structure {opponent_name: (mean,std)}
        data = {k:v[0] for k,v in data.items()} #extract only the mean
    except FileNotFoundError: 
        print("No data file found")
    return data
