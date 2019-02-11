import pytest
from unittest.mock import Mock
from rl_base.sarsa_agent import SarsaAgent
from functools import reduce

@pytest.fixture
def agent():
    agent = SarsaAgent()
    side_effect = lambda a: reduce(lambda x,y: x + str(y), a, "")
    agent._state_to_string = Mock(side_effect=side_effect)
    side_effect = lambda a: a
    agent._get_available_actions = Mock(side_effect=side_effect)
    return agent

def _perform_select_action_multiple_times(agent, state, times):
    actions = []
    for _ in range(times):
        actions.append(agent.select_action(state))
    return actions

def test_select_action_before_learn(agent):
    actions = _perform_select_action_multiple_times(agent, [1, 2], 20)
    assert actions.count(1) > 0
    assert actions.count(2) > 0

# test case:
# step 1: state = [1, 2], reward = 1 and 2 respectively
# step 2: state = [10, 30] or [20, 40], reward = 10, 30, 20 and 40 respectively
def test_sarsa_with_greedy(agent):
    agent.set_epsilon(0)

    # try 1-30
    agent.learn([1, 2], 1, 1, [10, 30], 30)

    assert agent.select_action([1, 2]) == 1

    agent.learn([10, 30], 30, 30, [], None)

    assert agent.select_action([1, 2]) == 1
    assert agent.select_action([10, 30]) == 30

    # try 2-20
    agent.learn([1, 2], 2, 2, [20, 40], 20)
    agent.learn([20, 40], 20, 20, [], None)

    # it picks 2 because Q([1,2],1) was calculated with Q([10,30],30) = 0.
    assert agent.select_action([1, 2]) == 2
    assert agent.select_action([20, 40]) == 20

    # run the learns again to update Q([1,2],1) and Q([1,2],2)
    agent.learn([1, 2], 1, 1, [10, 30], 30)
    agent.learn([1, 2], 2, 2, [20, 40], 20)

    # now it picks 1
    assert agent.select_action([1, 2]) == 1

    # try 40
    agent.learn([20, 40], 40, 40, [], None)

    # it still picks 1 because Q([1,2],2) was calculated with Q([20,40],20)
    assert agent.select_action([1, 2]) == 1
    assert agent.select_action([20, 40]) == 40

    # run the learn with 2 and 40 to update Q([1,2],2) with Q([20,40],40)
    agent.learn([1, 2], 2, 2, [20, 40], 40)

    # now it picks 2
    assert agent.select_action([1, 2]) == 2

def _assert_by_running_select_action_multiple_times(agent, state, times, more_probable_value):
    if len(state) != 2:
        raise ValueError("The state must be a list with two elements.")

    actions = _perform_select_action_multiple_times(agent, state, times)
    less_probable_value = state[1] if state[0] == more_probable_value else state[0]
    count_larger = actions.count(more_probable_value)
    count_smaller = actions.count(less_probable_value)
    assert count_larger > count_smaller and count_smaller > 0

def test_sarsa_with_epsilon_greedy(agent):
    agent.set_epsilon(0.7)
    
    # try 1-30
    agent.learn([1, 2], 1, 1, [10, 30], 30)
    agent.learn([10, 30], 30, 30, [], None)

    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 1)
    _assert_by_running_select_action_multiple_times(agent, [10, 30], 100, 30)

    # try 2-20
    agent.learn([1, 2], 2, 2, [20, 40], 20)
    agent.learn([20, 40], 20, 20, [], None)

    # it picks 2 more times because Q([1,2],1) was calculated with Q([10,30],30) = 0.
    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 2)
    _assert_by_running_select_action_multiple_times(agent, [20, 40], 100, 20)

    # run the learns again to update Q([1,2],1) and Q([1,2],2)
    agent.learn([1, 2], 1, 1, [10, 30], 30)
    agent.learn([1, 2], 2, 2, [20, 40], 20)

    # now it picks 1
    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 1)

    # try 40
    agent.learn([20, 40], 40, 40, [], None)

    # it still picks 1 because Q([1,2],2) was calculated with Q([20,40],20)
    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 1)
    _assert_by_running_select_action_multiple_times(agent, [20, 40], 100, 40)

    # run the learn with 2 and 40 to update Q([1,2],2) with Q([20,40],40)
    agent.learn([1, 2], 2, 2, [20, 40], 40)

    # now it picks 2
    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 2)

    # keep the count of 2 with this epsilon for the next text
    actions = _perform_select_action_multiple_times(agent, [1, 2], 100)
    count2_larger_e = actions.count(2)

    # with a smaller epsilon
    agent.set_epsilon(0.2)
    _assert_by_running_select_action_multiple_times(agent, [1, 2], 100, 2)
    actions = _perform_select_action_multiple_times(agent, [1, 2], 100)
    count2_smaller_e = actions.count(2)
    assert count2_smaller_e > count2_larger_e