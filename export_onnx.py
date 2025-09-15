import torch as th
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

# Export to ONNX for embedding into ProtoTwin models using ONNX Runtime Web
def export():
    class OnnxableSB3Policy(th.nn.Module):
        def __init__(self, policy: BasePolicy):
            super().__init__()
            self.policy = policy # Returns actions, values, log_probs

        # we create this method to match the expected signature for ONNX export
        def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
            return self.policy(observation, deterministic=True)
    
    # Load the trained ML model
    model = PPO.load("model-v2", device="cpu")

    # Create the Onnx policy
    # the model.policy is the policy network of the trained model
    onnx_policy = OnnxableSB3Policy(model.policy)

    # model.observation_space.shape gives the shape of the observation space
    observation_size = model.observation_space.shape
    # Create a dummy input with the correct shape
    # we create a batch of 1 observation
    # th.randn generates a tensor with random values from a normal distribution
    # we use this dummy input to trace the model's computation graph during export
    # without it, the export would not know the input shape
    # it is like an example input so that th.onnx.export can understand the model's structure
    dummy_input = th.randn(1, *observation_size)
    # opset_version=17 is the version of the ONNX operator set to use
    # onnx_policy is the model to be exported
    th.onnx.export(onnx_policy, dummy_input, "cartpole-v2.onnx", opset_version=17, input_names=["input"], output_names=["output"])

export()