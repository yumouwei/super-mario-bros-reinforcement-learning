import warnings
warnings.filterwarnings('ignore')

from gym_utils import SMBRamWrapper, load_smb_env, SMB
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY 
from gym_utils import CustomJoypadSpace

# With apply_api_compatibility can't find env.unwrapped.ram
# env_old = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env_wrap = SMBRamWrapper(env)
# env_wrap.reset()

# env2 = DummyVecEnv([lambda: env])
# env2.reset()


env_wrap2 = DummyVecEnv([lambda: env_wrap])
a = env_wrap2.reset()