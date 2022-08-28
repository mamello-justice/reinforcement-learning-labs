###
# Group Members
# Mamello:1851317
# Name:Student Number
# Name:Student Number
# Name:Student Number
###

import numpy as np
from environments.gridworld import GridworldEnv
import inspect
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def print_trajectory(trajectory):
    output = ""
    for row in trajectory:
        row_output = ""
        for action in row:
            if action == -1:
                row_output += " o "
            elif action == 0:
                row_output += " U "
            elif action == 1:
                row_output += " R "
            elif action == 2:
                row_output += " D "
            else:
                row_output += " L "

        output += row_output.strip() + "\n"
    output = output[:-2] + "X"      
    print(output)

def transformed_transition_dynamics(env):
    """
    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
            
    Returns:
        A tuple (prob, next_state, reward) of (env.observation_space.n x env.action_space.n) matrices
        prob: represents deterministic probabilities
        next_state: represents new states when taking action on a particular state
        reward: represents reward when taking action on a particular state
    """
    num_states = env.observation_space.n
    
    # Drop done axis and convert to NDArray
    P = np.array([[x[0][:-1] for x in env.P[s].values()] for s in range(num_states)])
    
    prob = P[:, :, 0].astype(dtype=np.float32)
    next_state = P[:, :, 1].astype(dtype=np.uint8)
    reward = P[:, :, 2].astype(dtype=np.float32)
        
    return prob, next_state, reward

def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    num_states = env.observation_space.n
    
    prob, next_state, reward = transformed_transition_dynamics(env)
    
    # Initialize V to zeros    
    V = np.zeros(num_states, dtype=np.float32)

    while True:
        delta = 0.0

        for s in range(num_states):
            v = V[s]
            
            V[s] = (policy[s] * prob[s]).dot(reward[s] + discount_factor * V[next_state[s]])
            
            delta = np.maximum(delta, abs(v - V[s]))

        if delta < theta:
            break
    return V

def random_policy(shape: tuple[int, int]):
    random_values = np.random.randint(1, 5, shape)
    sums = np.tile(random_values.sum(axis=1), (shape[1], 1)).transpose()
    return random_values / sums

def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    prob, next_state, reward = transformed_transition_dynamics(env)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        return prob[state] * (reward[state] + discount_factor * V[next_state[state]])
    
    
    # Initialize V[s] to zeros
    V = np.zeros(num_states)
    
    # Initialize policy randomly
    policy = random_policy((num_states, num_actions))
    
    while True:
        # Policy evaluation
        V = policy_evaluation_fn(env, policy, discount_factor)
        
        # Policy improvement
        policy_stable = True
        
        for s in range(num_states):
            action = np.argmax(policy[s])
            values = one_step_lookahead(s, V)
            max_action = values == np.max(values)
            policy[s] = np.zeros(num_actions)
            policy[s, max_action] = 1 / np.sum(max_action)
            
            if not max_action[action]:
                policy_stable = False
        
        if policy_stable:
            break

    return policy, V

POLICY_ITERATION_SETUP = f'''
import numpy as np
from environments.gridworld import GridworldEnv

{inspect.getsource(transformed_transition_dynamics)}

{inspect.getsource(random_policy)}

{inspect.getsource(policy_evaluation)}

{inspect.getsource(policy_iteration)}

env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
'''


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    prob, next_state, reward = transformed_transition_dynamics(env)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        return prob[state] * (reward[state] + discount_factor * V[next_state[state]])
    
    
    # Initialize V[s] to zeros
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        
        for s in range(num_states):
            v = V[s]
            
            V[s] = np.max(one_step_lookahead(s, V))
            
            delta = np.maximum(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    policy = np.zeros((num_states, num_actions))
    for s in range(num_states):
        max_action = np.argmax(one_step_lookahead(s, V))
        policy[s, max_action] = 1

    return policy, V


VALUE_ITERATION_SETUP = f'''
import numpy as np
from environments.gridworld import GridworldEnv

{inspect.getsource(transformed_transition_dynamics)}

{inspect.getsource(value_iteration)}

env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
'''

def main():
    print("*" * 5 + " Random policy trajectory " + "*" * 5)
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy = np.full((num_states, num_actions), 1 / num_actions)
    trajectory = np.full((num_states), -1)
    while True:
        action = np.random.choice(np.arange(num_actions), size=1, replace=False, p=policy[state])[0]
        
        trajectory[state] = action
        state, reward, done, _ = env.step(action)

        if done:
            break

    print_trajectory(trajectory.reshape(env.shape))
    print("")

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    v = policy_evaluation(env, policy)
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    
    policy, v = policy_iteration(env)
    print_trajectory(policy.reshape((env.shape[0], env.shape[1], -1)).argmax(axis=2))
    print("")
    print(v.reshape(env.shape))
    print("")
    
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    
    policy, v = value_iteration(env)
    print_trajectory(policy.reshape((env.shape[0], env.shape[1], -1)).argmax(axis=2))
    print("")
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)
    
    discount_factors = np.logspace(-0.2, 0, num=30)
    runs = 10
    
    policy_iteration_times = np.zeros(len(discount_factors))
    value_iteration_times = np.zeros(len(discount_factors))
    
    for i, discount in enumerate(discount_factors):
        policy_iteration_times[i] = np.mean(timeit.repeat(setup = POLICY_ITERATION_SETUP,
                        stmt = f"policy_iteration(env = env, discount_factor = {discount})",
                        repeat = runs,
                        number = 1))
        
        value_iteration_times[i] = np.mean(timeit.repeat(setup = VALUE_ITERATION_SETUP,
                        stmt = f"value_iteration(env = env, discount_factor = {discount})",
                        repeat = runs,
                        number = 1))
        
    width = 0.005
    
    fig, ax = plt.subplots()
    ax.bar(discount_factors - width / 2, policy_iteration_times, width = width, label = "Policy Iteration")
    ax.bar(discount_factors + width / 2, value_iteration_times, width = width, label = "Value Iteration")
    
    ax.set_title('Average time taken by policy and value iteration for different discount factors')
    ax.set_ylabel('Average time (seconds)')
    ax.set_xlabel('Discount factors')
    ax.set_xticks(discount_factors)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend()
    
    plt.show()
    

if __name__ == "__main__":
    main()
