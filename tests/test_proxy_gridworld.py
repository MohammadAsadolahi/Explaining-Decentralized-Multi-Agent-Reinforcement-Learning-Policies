from dmarl_explain.collector import collect_episodes
from dmarl_explain.proxy_gridworld import TaskGridworld, LocalWaypointPolicy, task_event_extractor
from dmarl_explain import build_hasse, assert_faithful_hasse


def test_actual_grid_state_action_reward_rollout_seeded():
    def collect():
        return collect_episodes(lambda: TaskGridworld(4, 7),
                                LocalWaypointPolicy(seed=5,exploration=.12),
                                task_event_extractor,num_episodes=12,seed=2026,
                                max_steps=2000)
    first, second = collect(), collect()
    assert first == second
    for trace in first:
        assert len({t for tasks in trace.values() for t in tasks}) == 7
        assert all(len(seq) == len(set(seq)) for seq in trace.values())
        assert_faithful_hasse(build_hasse(trace),trace)


def test_all_agent_members_share_joint_completion_event():
    env = TaskGridworld(2, 1,grid_size=4)
    env.reset(seed=2)
    t = 't0'
    # Both agents required if joint (otherwise at least the sole participant).
    members=env.participants[t]
    for m in members:env.positions[m]=env.goals[t]
    obs,rewards,term,trunc,infos=env.step({a:0 for a in env.routes})
    assert term and not trunc
    assert {a for a,v in infos['events'].items() if t in v} == members
    assert all(rewards[a] == 1.0 for a in members)
