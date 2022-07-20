from mprnn import utils,training,testing,baselines,compare, mp_env
from mprnn.sspagent import a2c_custom, advanced_rnn
from gym.envs.registration import register

register(
    id='mp-v0',
    entry_point='mprnn.mp_env.envs:MPEnv',
)