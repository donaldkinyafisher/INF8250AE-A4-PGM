from agents import ReinforcePolicy
from jax import numpy as jnp
from utils import Logger, Transition


def test_returns(transitions, discount): 

    discount_r = ReinforcePolicy.compute_discounted_returns(transitions, discount)

    return discount_r

if  __name__ == '__main__':

    dones = jnp.array([[False, False, True, False, False, False, True, False, True],
                      [False, False, True, False, False, False, False, False, True]])
    rewards = jnp.array([[-1,-1,100,-1,100,-1,-1,-1,100,-1,100],
                        [-1,-1,100,-1,100,-1,100,-1,-1,-1,100]])
    observation = jnp.zeros((2, 9))
    actions = jnp.ones((2, 9))

    transition = Transition
    transition.observation = observation
    transition.reward = rewards
    transition.action = actions
    transition.done = dones
    transition.observation = observation

    test_returns(transition, 0.95)