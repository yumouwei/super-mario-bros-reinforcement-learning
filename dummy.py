from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv

# Custom wrapper to handle both `seed` and `options` arguments
class CustomJoypadSpace(JoypadSpace):
    def reset(self, seed=None, options=None, **kwargs):
        """
        Overrides the `reset()` method to ignore `seed` and `options` arguments.
        """
        return super().reset(**kwargs)
    
# Create the environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# Simplify the controls
env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)
# Using GrayScaleWrapper to pass image in grayscale
env = GrayScaleObservation(env, keep_dim=True)
# Wrap inside the Dummy Environment 
env = DummyVecEnv([lambda: env])

# Test the environment
state = env.reset()

state.shape         
print(f"State shape: {state.shape}")