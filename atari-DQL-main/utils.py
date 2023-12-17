import torch
import numpy as np
import gymnasium as gym

def convert_observation(observation):
    # Convierte la observación de un arreglo numpy a un tensor torch.
    return torch.from_numpy(np.array(observation))


class NoopStart(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)

        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

def wrap_env(env: gym.Env):
    """
    Preparar un entorno de Gym para entrenar o probar con un agente Atari.
    
    Args:
        env (gym.Env): El entorno de Gym original que se envolverá.    
    Returns:
        gym.Env: El entorno de Gym Wrapped.
    """

    # Convertir observaciones a escala de grises
    env = gym.wrappers.GrayScaleObservation(env)
    
    # Reescala las observaciones a una resolución más pequeña
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    
    # Pila de múltiples fotogramas consecutivos para proporcionar información temporal al agente
    env = gym.wrappers.FrameStack(env, num_stack=4)

    return env
