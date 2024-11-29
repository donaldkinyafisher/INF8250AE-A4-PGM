from agents import ReinforcePolicy
from jax import numpy as jnp
import jax
import equinox as eqx
from utils import Logger, Transition

def compute_loss(self, model_parameters: tuple[eqx.Module, eqx.Module], transitions: Transition) -> tuple[float, dict[str, float]]:
    
    actor_parameters, critic_parameters = model_parameters
    ### ------------------------- To implement -------------------------
    observation = transitions.observation
    
    if observation.ndim or transitions.next_observation.ndim== 4:  
    # Batched case
        observation = observation[0]
        next_observation = transitions.next_observation[0]

    predicted_values = self.critic.get_logits(critic_parameters, observation)  # V(s)
    next_values = self.critic.get_logits(critic_parameters, next_observation)  # V(s')

    # Compute target values: r + γ * V(s') * (1 - done)
    discount_factor = self.discount_factor  # Assuming discount_factor is a class attribute
    target_values = transitions.reward + discount_factor * next_values * (1.0 - transitions.done)

    # Critic loss: Mean squared error between predicted and target values
    critic_loss = jnp.mean((predicted_values - jax.lax.stop_gradient(target_values)) ** 2)

    # Compute action probabilities and log probabilities
    logits = self.actor.get_logits(actor_parameters, observation)
    log_probs = jax.nn.log_softmax(logits, axis=-1)  # Log-probabilities of each action

    # Select log probabilities of the taken actions
    taken_action_log_probs = log_probs[jnp.arange(log_probs.shape[0]), transitions.action]

    # Compute advantages: A(s, a) = r + γV(s') - V(s)
    advantages = jax.lax.stop_gradient(target_values - predicted_values)

    # Actor loss: Negative of the advantage-weighted log probabilities
    actor_loss = -jnp.mean(taken_action_log_probs * advantages)

    # Total loss: Combine actor and critic losses
    loss = actor_loss + critic_loss
    ### ----------------------------------------------------------------

    loss_dict = {
        "Actor loss": actor_loss,
        "Critic loss": critic_loss
    }
    return loss, loss_dict


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