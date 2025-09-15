# STEP 1: Import dependencies
import asyncio
import os
import numpy as np
import math
import gymnasium
import prototwin
from prototwin_gymnasium import VecEnvInstance, VecEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# STEP 2: Define signal addresses (obtain these values from ProtoTwin)
cart_motor_target_velocity = 3
cart_motor_current_position = 5
cart_motor_current_velocity = 6
cart_motor_current_force = 7
pole_motor_current_position = 12
pole_motor_current_velocity = 13

# STEP 3: Create your vectorized instance environment by extending the base environment
class CartPoleEnv(VecEnvInstance):
    # Here self refers to the instance of CartPoleEnv
    # It is python's way of defining class methods
    def __init__(self, client: prototwin.Client, instance: int) -> None:
        super().__init__(client, instance)
        self.max_cart_position = 0.65  # Maximum cart distance
        self.duration = 10 # Truncation time in seconds

    # Normalization is a common technique in RL to scale values to a standard range
    # This helps the agent learn better and faster
    # e.g. position_val = -0.65 --> normalized_position = -1
    def normalize_position(self, position_val):
        '''
        Normalizes the cart's position to be within the range [-1, 1]
        '''
        return position_val / self.max_cart_position

    # e.g. angle_val = -pi --> normalized_angle = -1
    def normalize_angle(self, angle_val):
        '''
        Normalizes the pole's angle to be within the range [-1, 1]
        Using atan2(y, x) to get the radian angle and then normalizing it by pi as the limit is [-pi, pi]
        '''
        return math.atan2(math.sin(angle_val), math.cos(math.pi - angle_val)) / math.pi

    def reward_angle(self, normalized_angle):
        '''
        Reward for keeping the pole upright.
        '''
        return 1 - math.fabs(normalized_angle) # fabs means floating-point absolute value

    def reward_distance(self, normalized_position):
        '''
        Reward for keeping the cart close to the center.
        '''
        # Max reward is 1 when at center, decreases to 0 at limits
        # (e.g. 1 - 0 = 1 --> center, 1 - 1 = 0 --> limit).
        # Only writing `return math.fabs(normalized_position)` will give negative rewards (penalty)
        # (e.g. 1 - 1 = 0 --> center, 1 - 0 = 1 --> limit).
        return 1 - math.fabs(normalized_position)

    def reward_force_penalty(self):
        '''
        Penalize for using excessive force.
        '''
        return math.fabs(self.get(cart_motor_current_force))

    def reward(self, obs):
        '''
        Calculate the total reward for the current state.
        The hyperparameters here can be tuned for the desired behavior.
        '''
        # obs[0] and obs[1] comes from self.observations() function below
        # obs[0] is normalized_cart_position
        # obs[1] is normalized_pole_angle
        dt = 0.01
        distance_reward = self.reward_distance(obs[0])
        angle_reward = self.reward_angle(obs[1])
        force_penalty = self.reward_force_penalty()

        total_reward = 0.8 * angle_reward + 0.2 * distance_reward  - 0.004 * force_penalty
        return max(total_reward * dt, 0) # Ensure reward is non-negative. But we can improve it by not 
                                         # using the max function as we will see in main-v1.py

    def observations(self):
        '''
        Get the current observations from the environment
        '''
        cart_position = self.get(cart_motor_current_position) # Read the current position of the cart
        cart_velocity = self.get(cart_motor_current_velocity) # Read the current velocity of the cart
        pole_angular_distance = self.get(pole_motor_current_position) # Read the current angular distance of the pole [outputs in radians]
        pole_angular_velocity = self.get(pole_motor_current_velocity) # Read the current angular velocity of the pole [outputs in radians]

        # Normalize position and angle for the observation space
        normalized_cart_position = self.normalize_position(cart_position)
        normalized_pole_angle = self.normalize_angle(pole_angular_distance)

        # Return the observations as a numpy array
        # The order of observations is important and should match the training setup
        # By that I mean the order in which the observations are defined in the observation space
        return np.array([normalized_cart_position, normalized_pole_angle, cart_velocity, pole_angular_velocity])

    def reset(self, seed = None):
        super().reset(seed=seed)
        return self.observations(), {} # Return initial observation and empty info. 
                                       # Do not remove the {} as it is required by the Gymnasium API

    def apply(self, action):
        '''
        Apply the given action to the environment.
        '''
        # .set(signal_address, value) stores the value to the given signal address
        # action[0] means the first element of the action array
        self.set(cart_motor_target_velocity, action[0]) # Apply action by setting the cart's target velocity

    def step(self):
        obs = self.observations()
        reward = self.reward(obs) # Calculate reward
        done = abs(obs[0]) > 1 # Terminate if cart goes beyond limits
        truncated = self.time > self.duration # Truncate after the defined duration
        return obs, reward, done, truncated, {}

# STEP 4: Setup the training session
async def main():

    # Launch ProtoTwin Connect and load the cartpole-v1 model
    client = await prototwin.start()
    filepath = os.path.join(os.path.dirname(__file__), "cartpole-v0.ptm")
    await client.load(filepath)

    # The observation space contains:
    # 0. A measure of the cart's distance from the center, where 0 is at the center and +/-1 is at the limit
    # 1. A measure of the pole's angular distance from the upright position, where 0 is at the upright position 
    #    and +/-1 is at the down position
    # 2. The cart's current velocity (m/s)
    # 3. The pole's angular velocity (rad/s)

    # The explanation of dtype is provided in the pdf document. See readme for link.
    # The array will look like --> np.array[1, 1, infinity, infinity]
    observation_high = np.array([1, 1, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
    # gymnasiuam.spaces.Box is used to define continuous spaces between a lower and upper bound
    observation_space = gymnasium.spaces.Box(-observation_high, observation_high, dtype=np.float32)
    
    # The action space contains only the cart's target velocity
    # because we are controlling the cart motor by setting its target velocity
    # Output values will be in the range [-1.0, 1.0]
    # That is why we set [-1.0, 1.0] on ProtoTwin editor using Typescript
    # Such range is standard practice for normalizing continuous actions like velocity or force.
    action_high = np.array([1.0], dtype=np.float32)
    action_space = gymnasium.spaces.Box(-action_high, action_high, dtype=np.float32)

    # Create the vectorized environment
    entity_name = "MAIN-v0"
    num_envs = 64 # Number of parallel environments
    # pattern and spacing are optional parameters and default to prototwin.Pattern.GRID and 1 respectively
    env = VecEnv(CartPoleEnv, client, entity_name, num_envs, observation_space, action_space)
    env = VecMonitor(env) # Monitor the training progress

    # Create callback to regularly save the model
    save_freq = 10000 # Number of timesteps per instance
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./logs-v0/checkpoints/",
        name_prefix="checkpoint", save_replay_buffer=True, save_vecnormalize=True)

    # Define learning rate schedule. Described in the tldraw link. See readme for link.
    def lr_schedule(progress_remaining):
        initial_lr = 0.003
        return initial_lr * (progress_remaining ** 2) # ** means "to the power of"

    # Define the ML model
    # Change device to "cpu" if you do not have a CUDA-capable GPU
    model = PPO(MlpPolicy, env, device="cuda", verbose=1,
                batch_size=4096, n_steps=1000, learning_rate=lr_schedule, tensorboard_log="./tensorboard-v0/")

    # Start training!
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

if __name__ == '__main__':
    asyncio.run(main())