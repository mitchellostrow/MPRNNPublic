from pathlib import Path

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

