import os
from pathlib import Path
import json
from mprnn.utils import convert_dist_to_params, train_env_json, nn_json, train_net, FILEPATH
import numpy as np

rng = np.random.default_rng()
def train_net(model,steps,name="",loc=None):
    '''
    Trains, saves, and returns model
    Inputs:
        model (stable-baselines): untrained
        steps (counting number): number of training steps
        name (str): optional name for the model
        loc (str): directory in which to save the model
    Outputs:
        model: trained model
    '''
    print("MODEL IS:", model,name,loc)
    model.learn(total_timesteps = steps,log_interval=100,tb_log_name="test_run")
    if loc is None:
        model.save(f"model{name}.zip")
    else:
        model.save(Path(loc,f"model{name}.zip"))
    return model
if __name__ == "__main__":
    with open(Path(FILEPATH,"data","train_params.json")) as f:
        script_kwargs = json.load(f)
    it = 0
    try:
        os.mkdir(f"run{it}")
    except FileExistsError:
        #make folder for the model
        while True:
            try:
                os.mkdir(f"run{it}")
                break
            except FileExistsError:
                it += 1
                continue
    os.chdir(f"run{it}")    
    with open(f'train_params.json', 'w', encoding='utf-8') as f:
        #save this instance of the training parameters so we can refer back
        json.dump(script_kwargs, f, ensure_ascii=False, indent=4)
    rng.seed(script_kwargs['seed'])

    train_params = convert_dist_to_params(script_kwargs)
    
    for name in ["SSP","notSSP"]: #looping so we can directly compare in the same folder/clustering plot
        os.mkdir(name)
        train_params['net_params']["SSP"] = name == "SSP"
    
        env = train_env_json(train_params['env'],useparams=True)    
        model = nn_json(env,train_params)
        trainiters = [0,*[1000]*8]
        totaliters = 0
        for it in trainiters:
            print(it)
            totaliters += it
            time = str(totaliters)+"k"
            loc = f"{name}/{time}"
            os.mkdir(loc)
            model = train_net(model,it*1000,name=time,loc=loc)
            