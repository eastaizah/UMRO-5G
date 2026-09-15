"""The hand-written gradients are the part of learners.py most likely to be wrong,
so they are checked against central finite differences.  If these tests pass, the
networks really are being trained by backpropagation."""

import numpy as np
import pytest

from learners import (MLP, Adam, ReplayBuffer, MultiReplayBuffer, RunningNorm,
                      MultiCellPowerEnv, QMIXAgent, GNNDQNAgent, DQNAgent,
                      FedAvgAgent, huber_grad, joint_index, joint_decode,
                      epsilon, convergence_diagnostics, AGENT_REGISTRY)
from config import RLENV, RLTRAIN


def _numeric_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        a = f()
        x[i] = old - eps
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
        it.iternext()
    return g


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 4], rng)
    x = rng.normal(0, 1, (6, 5))
    target = rng.normal(0, 1, (6, 4))

    def loss():
        return 0.5 * float(np.sum((net.forward(x, cache=False) - target) ** 2))

    out = net.forward(x)
    grads, dx = net.backward(out - target)
    for k in net.params:
        num = _numeric_grad(loss, net.params[k])
        assert np.allclose(grads[k], num, atol=1e-5), k


def test_mlp_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    net = MLP([4, 6, 3], rng)
    x = rng.normal(0, 1, (5, 4))
    target = rng.normal(0, 1, (5, 3))

    def loss():
        return 0.5 * float(np.sum((net.forward(x, cache=False) - target) ** 2))

    out = net.forward(x)
    _, dx = net.backward(out - target)
    num = _numeric_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-5)


def test_qmix_mixer_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    env = MultiCellPowerEnv(np.random.default_rng(3), RLENV)
    agent = QMIXAgent(rng, env, RLTRAIN)
    q_vec = rng.normal(0, 1, (4, agent.n_agents))
    s = rng.normal(0, 1, (4, env.state_dim))
    target = rng.normal(0, 1, 4)

    def loss():
        qt, _ = agent._mix_forward(q_vec, s, agent.mix)
        return 0.5 * float(np.sum((qt - target) ** 2))

    qt, cache = agent._mix_forward(q_vec, s, agent.mix)
    grads, dq = agent._mix_backward(qt - target, cache)
    for k in agent.mix:
        num = _numeric_grad(loss, agent.mix[k])
        assert np.allclose(grads[k], num, atol=1e-4), k
    num_q = _numeric_grad(loss, q_vec)
    assert np.allclose(dq, num_q, atol=1e-4)


def test_qmix_mixer_is_monotone_in_every_agent_utility():
    """The defining QMIX constraint: dQ_tot/dQ_i >= 0 for every agent i."""
    rng = np.random.default_rng(4)
    env = MultiCellPowerEnv(np.random.default_rng(5), RLENV)
    agent = QMIXAgent(rng, env, RLTRAIN)
    s = rng.normal(0, 1, (12, env.state_dim))
    q = rng.normal(0, 1, (12, agent.n_agents))
    base, _ = agent._mix_forward(q, s, agent.mix)
    for i in range(agent.n_agents):
        bumped = q.copy()
        bumped[:, i] += 0.3
        up, _ = agent._mix_forward(bumped, s, agent.mix)
        assert np.all(up >= base - 1e-9)


def test_gnn_gradients_match_finite_differences():
    rng = np.random.default_rng(6)
    env = MultiCellPowerEnv(np.random.default_rng(7), RLENV)
    agent = GNNDQNAgent(rng, env, RLTRAIN)
    x = rng.normal(0, 1, (3, env.n_cells, env.obs_dim))
    target = rng.normal(0, 1, (3, env.n_cells, agent.n_actions))

    def loss():
        return 0.5 * float(np.sum((agent._forward(x, agent.p, cache=False) - target) ** 2))

    q = agent._forward(x, agent.p, cache=True)
    grads = agent._backward(q - target)
    for k in agent.p:
        num = _numeric_grad(loss, agent.p[k])
        assert np.allclose(grads[k], num, atol=1e-4), k


def test_gnn_message_passing_actually_mixes_neighbours():
    rng = np.random.default_rng(8)
    env = MultiCellPowerEnv(np.random.default_rng(9), RLENV)
    agent = GNNDQNAgent(rng, env, RLTRAIN)
    x = np.zeros((1, env.n_cells, env.obs_dim))
    base = agent._forward(x, agent.p, cache=False)[0, 0].copy()
    x2 = x.copy()
    x2[0, 1, :] = 1.0                      # cell 1 is a neighbour of cell 0
    changed = agent._forward(x2, agent.p, cache=False)[0, 0]
    assert not np.allclose(base, changed)


def test_adam_reduces_a_quadratic_loss():
    rng = np.random.default_rng(10)
    net = MLP([3, 8, 2], rng)
    opt = Adam(net.params, 5e-2)
    x = rng.normal(0, 1, (32, 3))
    y = rng.normal(0, 1, (32, 2))
    first = last = None
    for i in range(200):
        out = net.forward(x)
        loss = 0.5 * float(np.mean((out - y) ** 2))
        if i == 0:
            first = loss
        last = loss
        g, _ = net.backward((out - y) / len(x))
        opt.step(net.params, g)
    assert last < 0.5 * first


def test_adam_gradient_clipping_bounds_the_step():
    rng = np.random.default_rng(11)
    p = {"w": np.zeros(4)}
    opt = Adam(p, 1.0)
    opt.step(p, {"w": np.full(4, 1e6)}, clip=1.0)
    assert np.all(np.abs(p["w"]) <= 1.0 + 1e-6)


def test_huber_gradient_is_clipped():
    assert huber_grad(np.array([-5.0, -0.3, 0.4, 9.0]))[0] == -1.0
    assert huber_grad(np.array([-5.0, -0.3, 0.4, 9.0]))[3] == 1.0
    assert huber_grad(np.array([0.4])) == pytest.approx(0.4)


def test_joint_index_and_decode_round_trip():
    for idx in range(3 ** 4):
        a = joint_decode(idx, 4, 3)
        assert joint_index(a, 3) == idx


def test_replay_buffer_wraps_and_samples():
    rng = np.random.default_rng(12)
    buf = ReplayBuffer(5, 3, rng)
    for i in range(9):
        buf.add(np.full(3, i), i % 2, float(i), np.full(3, i + 1), i == 8)
    assert buf.n == 5
    s, a, r, s2, d = buf.sample(4)
    assert s.shape == (4, 3) and a.shape == (4,)


def test_multi_replay_buffer_shapes():
    rng = np.random.default_rng(13)
    buf = MultiReplayBuffer(6, 4, 6, 24, rng)
    for i in range(8):
        buf.add(np.zeros((4, 6)), np.zeros(24), [0, 1, 2, 0], 1.0,
                np.zeros((4, 6)), np.zeros(24), False)
    o, s, a, r, o2, s2, d = buf.sample(3)
    assert o.shape == (3, 4, 6) and a.shape == (3, 4)


def test_running_norm_matches_numpy():
    rng = np.random.default_rng(14)
    x = rng.normal(3, 2, 500)
    n = RunningNorm()
    for v in x:
        n.update(float(v))
    assert n.mean == pytest.approx(x.mean())
    assert n.std == pytest.approx(x.std(ddof=1))


def test_epsilon_decays_and_saturates():
    assert epsilon(0, RLTRAIN) == pytest.approx(RLTRAIN.eps_start)
    assert epsilon(RLTRAIN.eps_decay_episodes, RLTRAIN) == pytest.approx(RLTRAIN.eps_end)
    assert epsilon(10 * RLTRAIN.eps_decay_episodes, RLTRAIN) == pytest.approx(RLTRAIN.eps_end)


def test_environment_is_reproducible_and_bounded():
    def rollout(seed):
        env = MultiCellPowerEnv(np.random.default_rng(seed), RLENV)
        env.reset()
        tot = 0.0
        done = False
        while not done:
            _, _, r, done, _ = env.step([1, 1, 1, 1])
            tot += r
        return tot
    assert rollout(21) == rollout(21)
    assert rollout(21) != rollout(22)


def test_observations_are_finite_and_clipped():
    env = MultiCellPowerEnv(np.random.default_rng(23), RLENV)
    obs, state = env.reset()
    assert obs.shape == (RLENV.n_cells, env.obs_dim)
    assert state.shape == (env.state_dim,)
    for _ in range(20):
        obs, state, r, done, info = env.step([0, 2, 1, 2])
        assert np.all(np.isfinite(obs)) and np.all(np.abs(obs) <= 8.0)
        assert np.isfinite(r)


def test_myopic_optimum_dominates_any_single_action():
    env = MultiCellPowerEnv(np.random.default_rng(24), RLENV)
    env.reset()
    best = env.myopic_optimum()
    for a in ([0] * 4, [1] * 4, [2] * 4, [0, 1, 2, 0]):
        probe = MultiCellPowerEnv(np.random.default_rng(24), RLENV)
        probe.reset()
        _, _, r, _, _ = probe.step(a)
        assert r <= best + 1e-6


def test_every_agent_produces_legal_actions_and_trains():
    env = MultiCellPowerEnv(np.random.default_rng(25), RLENV)
    for name, cls in AGENT_REGISTRY.items():
        e = MultiCellPowerEnv(np.random.default_rng(26), RLENV)
        agent = cls(np.random.default_rng(27), e, RLTRAIN)
        obs, state = e.reset()
        for _ in range(30):
            a = agent.act(obs, state, 0.2)
            assert len(a) == e.n_cells
            assert all(0 <= v < e.n_levels for v in a)
            obs2, state2, r, done, _ = e.step(a)
            agent.observe(obs, state, a, r, obs2, state2, done)
            agent.train_step()
            obs, state = obs2, state2


def test_fedavg_synchronises_the_clients():
    e = MultiCellPowerEnv(np.random.default_rng(28), RLENV)
    agent = FedAvgAgent(np.random.default_rng(29), e, RLTRAIN)
    assert not np.allclose(agent.qs[0].params["W0"], agent.qs[1].params["W0"])
    agent.end_episode(RLTRAIN.fedavg_period_episodes - 1)
    assert np.allclose(agent.qs[0].params["W0"], agent.qs[1].params["W0"])
    assert agent.n_rounds == 1


def test_convergence_diagnostics_flags_a_flat_improved_curve():
    curve = np.concatenate([np.linspace(0, 10, 40), np.full(60, 10.0)])
    d = convergence_diagnostics(curve, floor=0.0)
    assert d["converged"] is True
    assert d["episodes_to_95pct"] > 0


def test_convergence_diagnostics_flags_a_non_learner():
    rng = np.random.default_rng(30)
    curve = rng.normal(0.0, 0.05, 100)
    d = convergence_diagnostics(curve, floor=0.0)
    assert d["improved_over_random"] is False
    assert d["converged"] is False


def test_convergence_diagnostics_flags_a_still_rising_curve():
    curve = np.linspace(0, 50, 100)
    d = convergence_diagnostics(curve, floor=0.0)
    assert d["stable"] is False
    assert d["converged"] is False
