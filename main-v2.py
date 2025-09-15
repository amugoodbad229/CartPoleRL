# STEP 1: Import dependencies
import os
import asyncio
import random
import math
import torch
import numpy as np
import prototwin
import gymnasium
from prototwin_gymnasium import VecEnvInstance, VecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# STEP 2: Signal addresses (copy/paste these values from ProtoTwin)
cart_motor_target_position = 2
cart_motor_target_velocity = 3
cart_motor_current_position = 5
cart_motor_current_velocity = 6
cart_motor_current_force = 7
pole_motor_current_position = 12 # output pole angle
pole_motor_current_velocity = 13 # output pole angular velocity

# We used list variables here for easier scaling to more complex environments
# For example, if you had 2 cart motors and 2 pole motors, 
# you would just add their signal addresses to these lists
# This is the professional way of coding, instead of hardcoding values everywhere
# Why? because it is easier to maintain and modify the code later on
states = [cart_motor_current_position, pole_motor_current_position, 
          cart_motor_current_velocity, pole_motor_current_velocity]
actions = [cart_motor_target_position]

# We will use the previous action as part of the observation. 
# This helps the agent learn better.
# This is an upgrade from main-v1.py
state_size = len(states) # 4 states
action_size = len(actions) # 1 previous action
observation_size = state_size + action_size # 5 (4 states + previous action)

domain_randomization = True
max_cart_position = 0.65

# STEP 3: Create your own vectorized environment instance by extending the VecEnvInstance base class
class CartPoleEnv(VecEnvInstance):
    def __init__(self, client: prototwin.Client, instance: int):
        super().__init__(client, instance)
        self.duration = 10
        # [0] * action_size creates a list of zeros with length action_size
        # length of action_size is 1, so this creates [0]
        self.previous_action = [0] * action_size
        if domain_randomization:
            # .set_domain_randomization_params() is a custom function we created below
            self.set_domain_randomization_params()

    # We can improve main-v1.py by adding domain randomization
    # You can adjust these parameters to change the level of randomness
    # Higher values will make the task harder
    # In real life, encoders can have small misalignments and noise
    # Encoder means a sensor that measures the angle of the pole
    # So we want our agent to be robust to these variations
    def set_domain_randomization_params(self):
        '''
        Sets the domain randomization parameters
        '''
        # .encoder_constant_error is a constant error added to the pole angle
        # It simulates misalignment of the encoder
        # misalignment means that the encoder is not perfectly aligned with the pole
        # why use 0.25 degrees? because it is a small angle
        self.encoder_constant_error = random.uniform(math.radians(-0.25), math.radians(0.25))
        # .encoder_precision is the smallest change in angle that the encoder can detect
        # It simulates limited resolution of the encoder
        # why 2*pi? because there are 2*pi radians in a full circle
        # why divide by 600 to 2500? because a typical encoder has between 600 to 2500 pulses per revolution (PPR)
        # we randomly choose from a list of common encoder resolutions
        self.encoder_precision = (math.pi * 2.0) / random.choice([600, 800, 1000, 1200, 1500, 2000, 2500])
        # .encoder_max_noise is the maximum noise added to the pole angle
        # It simulates measurement noise from the encoder
        # why use 0.5 degrees? because it is a small angle
        self.encoder_max_noise = random.uniform(0, math.radians(0.5))
    
    def normalize_position(self, position):
        '''
        Normalizes the specified position so that it lies in the range [-1, 1]
        '''
        return position / max_cart_position

    def normalize_angle(self, angle):
        '''
        Normalizes the specified angle so that it lies in the range [-1, 1]
        '''
        return math.atan2(math.sin(angle), math.cos(math.pi - angle)) / math.pi

    # see pole_angle_observation() function below for explanation
    # e.g. round(0.123 / 0.01) will give us 12.3 which rounds to 12
    # then multiply by 0.01 gives us 0.12
    # so we rounded 0.123 to the nearest 0.01
    def round(self, value, base):
        '''
        Rounds the given value to the nearest multiple of the specified base.
        '''
        return round(value / base) * base

    # How pole_angle_observation() calls self.encoder_constant_error?
    # because it is a member variable of the class.
    # It is a concept from Object-Oriented Programming (OOP)
    # In OOP, we can access member variables without calling the function
    # That is the beauty of OOP. 
    # Simply put, a class is a blueprint where we define functions
    # and each variable is a building block
    # because they are part of the class instance (self)
    def pole_angle_observation(self):
        '''
        The angular distance of the pole from the upright position after domain randomization
        '''
        value = self.get(pole_motor_current_position) # Read the true angle of the pole
        if domain_randomization:
            value += self.encoder_constant_error # Add a constant error from the encoder
            value += random.uniform(-self.encoder_max_noise, self.encoder_max_noise) # Add some measurement noise from the encoder
            value = self.round(value, self.encoder_precision) # Round to the precision of the encoder
        value = self.normalize_angle(value)
        return value
    
    def reward_angle(self):
        '''
        Reward the agent for how close the angle of the pole is from the upright position
        '''
        pole_angle = self.pole_angle_observation()
        return 1 - math.fabs(pole_angle)
    
    def reward_position_penalty(self):
        '''
        Penalize the agent for moving the cart away from the center
        '''
        cart_position = self.get(cart_motor_current_position)
        return math.fabs(self.normalize_position(cart_position))
    
    def reward_force_penalty(self):
        '''
        Penalize the agent for high force in the cart's motor
        '''
        force = self.get(cart_motor_current_force)
        return math.fabs(force)
    
    # You can uncomment this function to enable velocity penalty
    # which encourages smoother movements
    '''
    def reward_velocity_penalty(self):
    
        # Penalize the agent for high linear and angular velocities to encourage smoothness.

        cart_velocity = self.get(cart_motor_current_velocity)
        pole_angular_velocity = self.get(pole_motor_current_velocity)
        
        # We use the square of the velocities to more strongly punish high speeds

        return (cart_velocity**2) + (pole_angular_velocity**2)
    '''


    def reward(self):
        '''
        Reward function
        '''
        dt = 0.005
        reward = 0
        reward += self.reward_angle()
        reward -= self.reward_position_penalty()  * 0.5
        reward -= self.reward_force_penalty() * 0.0025

        # reward -= self.reward_velocity_penalty() * 0.01 # Uncomment this line to enable velocity penalty
        return reward * dt
    
    def observations(self):
        '''
        The current observations
        '''
        obs = [0] * observation_size
        for i in range(state_size):
            obs[i] = self.get(states[i])
        for i in range(action_size):
            obs[state_size + i] = self.previous_action[i]
        obs[0] = self.normalize_position(self.get(cart_motor_current_position))
        obs[1] = self.pole_angle_observation()
        return np.array(obs)
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self.previous_action = [0] * action_size
        if domain_randomization:
            self.set_domain_randomization_params()
        return self.observations(), {}
    
    def apply(self, action):
        self.action = action
        for i in range(action_size):
            # When i = 0, this line becomes:
            # self.set(actions[0], action[0])
            # which is the same as:
            # self.set(cart_motor_target_position, action[0])
            # which is the same as:
            # self.set(2, action[0])
            self.set(actions[i], action[i])

    def step(self):
        obs = self.observations()
        self.previous_action = self.action
        reward = self.reward()
        done = abs(obs[0]) > 1
        truncated = self.time > self.duration
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():
    
    # Launch ProtoTwin Connect and load the cartpole-v2 model
    client = await prototwin.start()
    filepath = os.path.join(os.path.dirname(__file__), "cartpole-v2.ptm")
    await client.load(filepath)

    # Multiplication operator * on a list, it's a shortcut for repetition
    observation_high = np.array([np.finfo(np.float32).max] * observation_size, dtype=np.float32)
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)
    
    # Output values will be in the range [-max_cart_position, max_cart_position]
    # because we are controlling the target position of the cart motor
    # That is why we set [-0.65, 0.65] on ProtoTwin editor using Typescript
    action_high = np.array([max_cart_position] * action_size, dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    # Create callback to regularly save the model
    save_freq = 5000
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs-v2/checkpoints/", 
                                             name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule
    def lr_schedule(progress_remaining):
        initial_lr = 0.003
        return initial_lr * (progress_remaining ** 2)
    
    # Create the vectorized environment
    entity_name = "MAIN-v2"
    num_envs = 64
    pattern = prototwin.Pattern.GRID
    spacing = 1
    env = VecEnv(CartPoleEnv, client, entity_name, num_envs, observation_space, action_space, pattern=pattern, spacing=spacing)
    env = VecMonitor(env) # Monitor the training progress

    # Define the ML model
    batch_size = 10000
    n_steps = 1000
    ent_coef = 0.0001 # Entropy coefficient encourages exploration.

    # policy_kwargs allows us to customize the policy network architecture and value function network architecture.
    # Here, activation_fn is set to ReLU (Rectified Linear Unit) which is a common activation function.
    # We use ReLU because it helps the model learn complex patterns in the data.
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[128, 64, 32], vf=[128, 64, 32]))
    model = PPO(MlpPolicy, env, verbose=1, batch_size=batch_size, n_steps=n_steps, ent_coef=ent_coef, 
                policy_kwargs=policy_kwargs, learning_rate=lr_schedule, tensorboard_log="./tensorboard-v2/", device="cuda")

    # Start training!
    model.learn(total_timesteps=20_000_000, callback=checkpoint_callback)

if __name__ == '__main__':
    asyncio.run(main())