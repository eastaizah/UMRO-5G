"""
learners.py -- Reinforcement-learning controllers, implemented for real in NumPy.

WHAT IS ACTUALLY IMPLEMENTED (this list is reproduced verbatim in work/sim_results.md;
the article must not claim more than what is written here):

  * `MLP`          -- fully connected network, tanh hidden units, linear output, with
                      hand-written forward and backward passes and an Adam optimiser.
                      No autodiff framework is used; the gradients are derived and
                      coded explicitly and are checked against finite differences in
                      simulations/tests/test_learners.py.
  * `DQNAgent`     -- Deep Q-Network: uniform experience replay, a separate target
                      network refreshed every `target_update_steps` gradient steps,
                      epsilon-greedy exploration with linear decay, Huber loss, and
                      gradient clipping.  This is a full DQN, not a tabular proxy.
  * `QMIXAgent`    -- multi-agent DQN with one independent per-cell Q-network and a
                      QMIX monotonic mixing network whose weights are produced by
                      hypernetworks conditioned on the global state and forced
                      non-negative by an absolute value, which guarantees
                      d Q_tot / d Q_i >= 0 (the QMIX monotonicity condition).
                      Centralised training, decentralised execution (CTDE).
  * `FedAvgAgent`  -- one local DQN per cell, each with its own replay buffer, trained
                      only on local transitions; every `fedavg_period_episodes`
                      episodes the parameter vectors are averaged (uniform FedAvg,
                      equal client weights) and broadcast back.
  * `GNNDQNAgent`  -- node features per cell, `gnn_message_rounds` rounds of
                      normalised-adjacency message passing with shared self and
                      neighbour weight matrices, then a shared per-node Q head.
                      Trained by independent Q-learning with parameter sharing on the
                      global reward.  Backpropagation through the message passing is
                      hand-derived.

WHAT IS *NOT* IMPLEMENTED: convolutional or recurrent encoders, prioritised replay,
double/duelling DQN, attention-based mixers, or any form of transfer from published
hyper-parameters.  Convergence is *measured*, and an agent that does not converge is
reported as not converged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import RLENV, RLTRAIN, RLEnvConfig, RLTrainConfig, CHANNEL
from channel import path_loss_db, noise_power_dbm


# ==================================================================================
# 1. Neural building blocks
# ==================================================================================

class Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr: float,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray],
             clip: float | None = None) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            if clip is not None:
                n = np.linalg.norm(g)
                if n > clip:
                    g = g * (clip / (n + 1e-12))
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class MLP:
    """Feed-forward network with tanh hidden activations and a linear output layer.

    forward(x) caches the activations; backward(dout) returns the parameter gradients
    and the gradient with respect to the input (needed by QMIX and the GNN).
    """

    def __init__(self, dims: Sequence[int], rng: np.random.Generator,
                 out_scale: float = 1.0):
        self.dims = list(dims)
        self.params: Dict[str, np.ndarray] = {}
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            scale = np.sqrt(1.0 / fan_in)
            if i == len(dims) - 2:
                scale *= out_scale
            self.params[f"W{i}"] = rng.normal(0.0, scale, (dims[i], dims[i + 1]))
            self.params[f"b{i}"] = np.zeros(dims[i + 1])
        self.n_layers = len(dims) - 1
        self._cache: list = []

    def forward(self, x: np.ndarray, cache: bool = True) -> np.ndarray:
        a = x
        c = [x]
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = np.tanh(z) if i < self.n_layers - 1 else z
            c.append(a)
        if cache:
            self._cache = c
        return a

    def backward(self, dout: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        c = self._cache
        grads: Dict[str, np.ndarray] = {}
        d = dout
        for i in reversed(range(self.n_layers)):
            a_prev = c[i]
            grads[f"W{i}"] = a_prev.T @ d
            grads[f"b{i}"] = d.sum(axis=0)
            d = d @ self.params[f"W{i}"].T
            if i > 0:
                d = d * (1.0 - c[i] ** 2)          # tanh'
        return grads, d

    def copy_params(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, p: Dict[str, np.ndarray]) -> None:
        for k in self.params:
            self.params[k][...] = p[k]


def huber_grad(td: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """d/dq of the Huber loss with residual td = q - target."""
    return np.clip(td, -delta, delta)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, rng: np.random.Generator):
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.rng = rng

    def add(self, s, a, r, s2, done) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(done)
        self.ptr = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


class MultiReplayBuffer:
    """Replay of joint transitions: per-agent observations plus the global state."""

    def __init__(self, capacity: int, n_agents: int, obs_dim: int, state_dim: int,
                 rng: np.random.Generator):
        self.cap = capacity
        self.o = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.o2 = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, n_agents), dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.ptr = 0
        self.rng = rng

    def add(self, o, s, a, r, o2, s2, done) -> None:
        i = self.ptr
        self.o[i], self.s[i], self.a[i] = o, s, a
        self.r[i], self.o2[i], self.s2[i], self.d[i] = r, o2, s2, float(done)
        self.ptr = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.n, batch)
        return (self.o[idx], self.s[idx], self.a[idx], self.r[idx],
                self.o2[idx], self.s2[idx], self.d[idx])


# ==================================================================================
# 2. Environment: multi-cell downlink power control with per-user queues
# ==================================================================================

OBS_DIM = 6


class MultiCellPowerEnv:
    """Slotted multi-cell interference environment.

    Each cell picks one of `n_power_levels` transmit powers per step.  Users share
    their cell's bandwidth equally; the SINR of user u served by cell c is

        SINR = P_c A[c,u,c] / ( sum_{j != c} P_j A[c,u,j] + N0 B_share )

    Backlogs make the problem a genuine MDP: a power decision that starves a URLLC
    user now is punished for several steps afterwards, so a myopic policy is not
    optimal and there is something for an RL agent to learn.
    """

    def __init__(self, rng: np.random.Generator, cfg: RLEnvConfig = RLENV):
        self.rng = rng
        self.cfg = cfg
        self.n_cells = cfg.n_cells
        self.upc = cfg.n_users // cfg.n_cells
        self.n_levels = cfg.n_power_levels
        self.powers = np.linspace(cfg.p_max_w / cfg.n_power_levels,
                                  cfg.p_max_w, cfg.n_power_levels)
        self.b_share = cfg.bandwidth_hz / self.upc
        self.noise_w = 10.0 ** ((cfg.noise_dbm - 30.0) / 10.0) * (self.b_share / 1e6)
        self.dt_s = 1e-3
        self.obs_dim = OBS_DIM
        self.state_dim = OBS_DIM * self.n_cells
        # normalised adjacency (with self-loop) for the GNN
        A = np.zeros((self.n_cells, self.n_cells))
        for i, nb in enumerate(cfg.adjacency):
            for j in nb:
                A[i, j] = 1.0
        A = A + np.eye(self.n_cells)
        self.adj_norm = A / A.sum(axis=1, keepdims=True)
        self.reset()

    # -- geometry / large scale -----------------------------------------------------
    def _place(self) -> np.ndarray:
        """Return A[c, u, j]: linear channel gain from transmitter j to user u of
        cell c (large-scale only)."""
        cfg = self.cfg
        ring_r = 500.0
        ang = 2 * np.pi * np.arange(self.n_cells) / self.n_cells
        cells = np.stack([ring_r * np.cos(ang), ring_r * np.sin(ang)], axis=1)
        gains = np.zeros((self.n_cells, self.upc, self.n_cells))
        for c in range(self.n_cells):
            r = np.sqrt(self.rng.random(self.upc)) * 250.0 + 35.0
            th = self.rng.uniform(0, 2 * np.pi, self.upc)
            pos = cells[c] + np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
            for j in range(self.n_cells):
                d = np.linalg.norm(pos - cells[j], axis=1)
                sh = self.rng.normal(0.0, CHANNEL.shadowing_sigma_db, self.upc)
                pl = path_loss_db(d) + sh
                gains[c, :, j] = 10.0 ** (-pl / 10.0)
        return gains

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.large = self._place()
        self.fading = self.rng.exponential(1.0, (self.n_cells, self.upc, self.n_cells))
        self.backlog = np.zeros((self.n_cells, self.upc))
        self.prev_action = np.full(self.n_cells, self.n_levels - 1)
        self.active = (self.rng.random(self.n_cells) < 0.6).astype(float)
        self.t = 0
        self._last_rate = np.zeros((self.n_cells, self.upc))
        return self.observations(), self.global_state()

    def _advance_activity(self) -> None:
        u = self.rng.random(self.n_cells)
        on = self.active > 0.5
        flip_off = on & (u < self.cfg.burst_on_to_off)
        flip_on = (~on) & (u < self.cfg.burst_off_to_on)
        self.active = np.where(flip_off, 0.0, np.where(flip_on, 1.0, self.active))

    # -- dynamics -------------------------------------------------------------------
    def _gains(self) -> np.ndarray:
        return self.large * self.fading

    def _rates(self, powers: np.ndarray, gains: np.ndarray) -> np.ndarray:
        """powers: (n_cells,) -> rate bit/s per user, shape (n_cells, upc)."""
        rx = gains * powers[None, None, :]
        serving = np.take_along_axis(
            rx, np.arange(self.n_cells)[:, None, None].repeat(self.upc, 1), axis=2
        )[:, :, 0]
        interf = rx.sum(axis=2) - serving
        sinr = serving / (interf + self.noise_w)
        return self.b_share * np.log2(1.0 + sinr)

    def _reward_from(self, rates: np.ndarray, backlog: np.ndarray,
                     powers: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        served = np.minimum(backlog, rates * self.dt_s)
        new_backlog = backlog - served
        util = np.sum(np.log1p(served / 1000.0))
        viol = new_backlog[:, :self.cfg.urllc_users_per_cell] > self.cfg.urllc_queue_budget_bits
        energy = self.cfg.energy_cost_weight * float(powers.sum()) / self.cfg.p_max_w
        reward = util - self.cfg.urllc_penalty * float(viol.sum()) - energy
        return float(reward), new_backlog, served

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, float, bool, dict]:
        actions = np.asarray(actions, dtype=int)
        powers = self.powers[actions]
        gains = self._gains()
        rates = self._rates(powers, gains)
        # arrivals first, then service (arrivals in the same slot are servable)
        self.backlog += (self.active[:, None]
                         * self.rng.exponential(self.cfg.arrival_bits_mean,
                                                (self.n_cells, self.upc)))
        reward, self.backlog, served = self._reward_from(rates, self.backlog, powers)
        self.backlog = np.minimum(self.backlog, 50 * self.cfg.arrival_bits_mean)
        self._last_rate = rates
        self.prev_action = actions
        self.t += 1
        self._advance_activity()
        if self.t % self.cfg.coherence_steps == 0:
            self.fading = self.rng.exponential(
                1.0, (self.n_cells, self.upc, self.n_cells))
        done = self.t >= self.cfg.episode_steps
        info = {"served_bits": float(served.sum()),
                "mean_rate_mbps": float(rates.mean() / 1e6),
                "power_w": float(powers.sum())}
        return self.observations(), self.global_state(), reward, done, info

    def myopic_optimum(self) -> float:
        """Best single-step reward over all `n_levels ** n_cells` joint actions,
        evaluated on the *current* channel and backlog.  This is the normaliser used
        for the '% of optimum' figures; it is a myopic (one-step-greedy) reference
        that additionally has access to per-user instantaneous CSI, which the agents
        do not.  It is not the optimal value of the MDP, and the article must say so."""
        gains = self._gains()
        best = -np.inf
        n_joint = self.n_levels ** self.n_cells
        backlog = self.backlog + self.active[:, None] * self.cfg.arrival_bits_mean
        for j in range(n_joint):
            idx = []
            x = j
            for _ in range(self.n_cells):
                idx.append(x % self.n_levels)
                x //= self.n_levels
            powers = self.powers[np.array(idx)]
            r, _, _ = self._reward_from(self._rates(powers, gains), backlog, powers)
            best = max(best, r)
        return float(best)

    # -- features -------------------------------------------------------------------
    def observations(self) -> np.ndarray:
        gains = self._gains()
        own = np.stack([gains[c, :, c] for c in range(self.n_cells)])
        tot = gains.sum(axis=2)
        interf = tot - own
        obs = np.zeros((self.n_cells, OBS_DIM))
        obs[:, 0] = (np.log10(own.mean(axis=1) + 1e-18) + 13.0) / 3.0
        obs[:, 1] = (np.log10(interf.mean(axis=1) + 1e-18) + 13.0) / 3.0
        obs[:, 2] = self.backlog.mean(axis=1) / self.cfg.arrival_bits_mean
        obs[:, 3] = (self.backlog[:, :self.cfg.urllc_users_per_cell].mean(axis=1)
                     / self.cfg.urllc_queue_budget_bits)
        obs[:, 4] = self.prev_action / max(self.n_levels - 1, 1)
        obs[:, 5] = np.log10(own.mean(axis=1) / (interf.mean(axis=1) + 1e-18) + 1e-9)
        return np.clip(obs, -8.0, 8.0)

    def global_state(self) -> np.ndarray:
        return self.observations().reshape(-1)


def joint_index(actions: Sequence[int], n_levels: int) -> int:
    idx = 0
    mult = 1
    for a in actions:
        idx += int(a) * mult
        mult *= n_levels
    return idx


def joint_decode(index: int, n_cells: int, n_levels: int) -> List[int]:
    out = []
    for _ in range(n_cells):
        out.append(index % n_levels)
        index //= n_levels
    return out


# ==================================================================================
# 3. Agents
# ==================================================================================

class BaseAgent:
    name = "base"

    def act(self, obs: np.ndarray, state: np.ndarray, eps: float) -> List[int]:
        raise NotImplementedError

    def observe(self, obs, state, actions, reward, obs2, state2, done) -> None:
        raise NotImplementedError

    def train_step(self) -> None:
        pass

    def end_episode(self, ep: int) -> None:
        pass


class DQNAgent(BaseAgent):
    """Centralised DQN over the joint action space."""

    name = "DQN"

    def __init__(self, rng, env: MultiCellPowerEnv, tr: RLTrainConfig = RLTRAIN):
        self.rng = rng
        self.env = env
        self.tr = tr
        self.n_actions = env.n_levels ** env.n_cells
        dims = [env.state_dim, *tr.hidden, self.n_actions]
        self.q = MLP(dims, rng, out_scale=0.1)
        self.target = MLP(dims, rng, out_scale=0.1)
        self.target.set_params(self.q.copy_params())
        self.opt = Adam(self.q.params, tr.lr)
        self.buf = ReplayBuffer(tr.replay_capacity, env.state_dim, rng)
        self.steps = 0

    def act(self, obs, state, eps):
        if self.rng.random() < eps:
            a = int(self.rng.integers(self.n_actions))
        else:
            q = self.q.forward(state[None, :], cache=False)[0]
            a = int(np.argmax(q))
        return joint_decode(a, self.env.n_cells, self.env.n_levels)

    def observe(self, obs, state, actions, reward, obs2, state2, done):
        self.buf.add(state, joint_index(actions, self.env.n_levels),
                     reward, state2, done)

    def train_step(self):
        tr = self.tr
        if self.buf.n < tr.warmup_transitions:
            return
        self.steps += 1
        if self.steps % tr.train_every:
            return
        s, a, r, s2, d = self.buf.sample(tr.batch_size)
        q_next = self.target.forward(s2, cache=False).max(axis=1)
        target = r + tr.gamma * (1.0 - d) * q_next
        q = self.q.forward(s)
        pred = q[np.arange(len(a)), a]
        dout = np.zeros_like(q)
        dout[np.arange(len(a)), a] = huber_grad(pred - target) / len(a)
        grads, _ = self.q.backward(dout)
        self.opt.step(self.q.params, grads, tr.grad_clip)
        if self.steps % tr.target_update_steps == 0:
            self.target.set_params(self.q.copy_params())


class QMIXAgent(BaseAgent):
    """Independent per-agent Q-networks + QMIX monotonic mixer (CTDE)."""

    name = "MADRL-QMIX"

    def __init__(self, rng, env: MultiCellPowerEnv, tr: RLTrainConfig = RLTRAIN):
        self.rng, self.env, self.tr = rng, env, tr
        self.n_agents = env.n_cells
        self.n_actions = env.n_levels
        dims = [env.obs_dim, *tr.hidden, self.n_actions]
        self.qs = [MLP(dims, rng, out_scale=0.1) for _ in range(self.n_agents)]
        self.tqs = [MLP(dims, rng, out_scale=0.1) for _ in range(self.n_agents)]
        for q, t in zip(self.qs, self.tqs):
            t.set_params(q.copy_params())
        h = tr.mixer_hidden
        sd = env.state_dim
        sc = 0.3
        self.mix = {
            "hw1": rng.normal(0, sc / np.sqrt(sd), (sd, self.n_agents * h)),
            "hb1": rng.normal(0, sc / np.sqrt(sd), (sd, h)),
            "hw2": rng.normal(0, sc / np.sqrt(sd), (sd, h)),
            "hv": rng.normal(0, sc / np.sqrt(sd), (sd, 1)),
        }
        self.tmix = {k: v.copy() for k, v in self.mix.items()}
        self.opts = [Adam(q.params, tr.lr) for q in self.qs]
        self.mix_opt = Adam(self.mix, tr.lr)
        self.buf = MultiReplayBuffer(tr.replay_capacity, self.n_agents,
                                     env.obs_dim, env.state_dim, rng)
        self.steps = 0
        self.h = h

    # -- mixer forward / backward ---------------------------------------------------
    def _mix_forward(self, q_vec: np.ndarray, s: np.ndarray, mix: Dict[str, np.ndarray]):
        B = q_vec.shape[0]
        raw_w1 = s @ mix["hw1"]
        w1 = np.abs(raw_w1).reshape(B, self.n_agents, self.h)
        b1 = s @ mix["hb1"]
        raw_w2 = s @ mix["hw2"]
        w2 = np.abs(raw_w2)
        v = s @ mix["hv"]
        hid_pre = np.einsum("bi,bih->bh", q_vec, w1) + b1
        hid = np.where(hid_pre > 0, hid_pre, np.exp(np.clip(hid_pre, -30, 0)) - 1.0)  # ELU
        q_tot = (hid * w2).sum(axis=1, keepdims=True) + v
        cache = (q_vec, s, raw_w1, w1, raw_w2, w2, hid_pre, hid)
        return q_tot.ravel(), cache

    def _mix_backward(self, dqtot: np.ndarray, cache):
        q_vec, s, raw_w1, w1, raw_w2, w2, hid_pre, hid = cache
        B = q_vec.shape[0]
        d = dqtot[:, None]                               # (B,1)
        dv = d
        dhid = d * w2
        dw2 = d * hid
        delu = np.where(hid_pre > 0, 1.0, hid + 1.0)
        dhid_pre = dhid * delu
        db1 = dhid_pre
        dw1 = np.einsum("bi,bh->bih", q_vec, dhid_pre)
        dq_vec = np.einsum("bih,bh->bi", w1, dhid_pre)
        grads = {
            "hw1": s.T @ (dw1 * np.sign(raw_w1).reshape(B, self.n_agents, self.h)
                          ).reshape(B, -1),
            "hb1": s.T @ db1,
            "hw2": s.T @ (dw2 * np.sign(raw_w2)),
            "hv": s.T @ dv,
        }
        return grads, dq_vec

    def act(self, obs, state, eps):
        acts = []
        for i in range(self.n_agents):
            if self.rng.random() < eps:
                acts.append(int(self.rng.integers(self.n_actions)))
            else:
                q = self.qs[i].forward(obs[i][None, :], cache=False)[0]
                acts.append(int(np.argmax(q)))
        return acts

    def observe(self, obs, state, actions, reward, obs2, state2, done):
        self.buf.add(obs, state, actions, reward, obs2, state2, done)

    def train_step(self):
        tr = self.tr
        if self.buf.n < tr.warmup_transitions:
            return
        self.steps += 1
        if self.steps % tr.train_every:
            return
        o, s, a, r, o2, s2, d = self.buf.sample(tr.batch_size)
        B = len(r)
        q_sel = np.zeros((B, self.n_agents))
        outs = []
        for i in range(self.n_agents):
            qi = self.qs[i].forward(o[:, i, :])
            outs.append(qi)
            q_sel[:, i] = qi[np.arange(B), a[:, i]]
        q_tot, cache = self._mix_forward(q_sel, s, self.mix)

        q_next = np.zeros((B, self.n_agents))
        for i in range(self.n_agents):
            q_next[:, i] = self.tqs[i].forward(o2[:, i, :], cache=False).max(axis=1)
        q_tot_next, _ = self._mix_forward(q_next, s2, self.tmix)
        target = r + tr.gamma * (1.0 - d) * q_tot_next

        dqtot = huber_grad(q_tot - target) / B
        mgrads, dq_vec = self._mix_backward(dqtot, cache)
        self.mix_opt.step(self.mix, mgrads, tr.grad_clip)
        for i in range(self.n_agents):
            dout = np.zeros_like(outs[i])
            dout[np.arange(B), a[:, i]] = dq_vec[:, i]
            g, _ = self.qs[i].backward(dout)
            self.opts[i].step(self.qs[i].params, g, tr.grad_clip)
        if self.steps % tr.target_update_steps == 0:
            for q, t in zip(self.qs, self.tqs):
                t.set_params(q.copy_params())
            self.tmix = {k: v.copy() for k, v in self.mix.items()}


class FedAvgAgent(BaseAgent):
    """One local DQN per cell; uniform FedAvg of the parameters every P episodes."""

    name = "FedAvg-DQN"

    def __init__(self, rng, env: MultiCellPowerEnv, tr: RLTrainConfig = RLTRAIN):
        self.rng, self.env, self.tr = rng, env, tr
        self.n_agents = env.n_cells
        self.n_actions = env.n_levels
        dims = [env.obs_dim, *tr.hidden, self.n_actions]
        self.qs = [MLP(dims, rng, out_scale=0.1) for _ in range(self.n_agents)]
        self.tqs = [MLP(dims, rng, out_scale=0.1) for _ in range(self.n_agents)]
        for q, t in zip(self.qs, self.tqs):
            t.set_params(q.copy_params())
        self.opts = [Adam(q.params, tr.lr) for q in self.qs]
        self.bufs = [ReplayBuffer(tr.replay_capacity // self.n_agents,
                                  env.obs_dim, rng) for _ in range(self.n_agents)]
        self.steps = 0
        self.n_rounds = 0

    def act(self, obs, state, eps):
        acts = []
        for i in range(self.n_agents):
            if self.rng.random() < eps:
                acts.append(int(self.rng.integers(self.n_actions)))
            else:
                acts.append(int(np.argmax(self.qs[i].forward(obs[i][None, :],
                                                             cache=False)[0])))
        return acts

    def observe(self, obs, state, actions, reward, obs2, state2, done):
        # each client only sees its own observation and the shared global reward
        for i in range(self.n_agents):
            self.bufs[i].add(obs[i], actions[i], reward / self.n_agents,
                             obs2[i], done)

    def train_step(self):
        tr = self.tr
        if self.bufs[0].n < tr.warmup_transitions // self.n_agents:
            return
        self.steps += 1
        if self.steps % tr.train_every:
            return
        for i in range(self.n_agents):
            s, a, r, s2, d = self.bufs[i].sample(tr.batch_size)
            qn = self.tqs[i].forward(s2, cache=False).max(axis=1)
            target = r + tr.gamma * (1.0 - d) * qn
            q = self.qs[i].forward(s)
            pred = q[np.arange(len(a)), a]
            dout = np.zeros_like(q)
            dout[np.arange(len(a)), a] = huber_grad(pred - target) / len(a)
            g, _ = self.qs[i].backward(dout)
            self.opts[i].step(self.qs[i].params, g, tr.grad_clip)
        if self.steps % tr.target_update_steps == 0:
            for q, t in zip(self.qs, self.tqs):
                t.set_params(q.copy_params())

    def end_episode(self, ep):
        if (ep + 1) % self.tr.fedavg_period_episodes:
            return
        keys = self.qs[0].params.keys()
        avg = {k: np.mean([q.params[k] for q in self.qs], axis=0) for k in keys}
        for q in self.qs:
            q.set_params(avg)
        self.n_rounds += 1


class GNNDQNAgent(BaseAgent):
    """Message-passing value network with parameter sharing across cells."""

    name = "GNN-DRL"

    def __init__(self, rng, env: MultiCellPowerEnv, tr: RLTrainConfig = RLTRAIN):
        self.rng, self.env, self.tr = rng, env, tr
        self.n_agents = env.n_cells
        self.n_actions = env.n_levels
        self.R = tr.gnn_message_rounds
        H = tr.gnn_hidden
        F = env.obs_dim
        self.A = env.adj_norm
        sc = 0.6

        def mk(rng_, shape):
            return rng_.normal(0.0, sc / np.sqrt(shape[0]), shape)

        self.p: Dict[str, np.ndarray] = {}
        din = F
        for r in range(self.R):
            self.p[f"Ws{r}"] = mk(rng, (din, H))
            self.p[f"Wn{r}"] = mk(rng, (din, H))
            self.p[f"b{r}"] = np.zeros(H)
            din = H
        self.p["Wo"] = mk(rng, (H, self.n_actions)) * 0.1
        self.p["bo"] = np.zeros(self.n_actions)
        self.tp = {k: v.copy() for k, v in self.p.items()}
        self.opt = Adam(self.p, tr.lr)
        self.buf = MultiReplayBuffer(tr.replay_capacity, self.n_agents,
                                     env.obs_dim, env.state_dim, rng)
        self.steps = 0

    def _forward(self, X: np.ndarray, p: Dict[str, np.ndarray], cache: bool):
        """X: (B, n_agents, F) -> Q: (B, n_agents, n_actions)."""
        c = []
        h = X
        for r in range(self.R):
            m = np.einsum("ij,bjf->bif", self.A, h)
            z = h @ p[f"Ws{r}"] + m @ p[f"Wn{r}"] + p[f"b{r}"]
            h_new = np.tanh(z)
            if cache:
                c.append((h, m, h_new))
            h = h_new
        q = h @ p["Wo"] + p["bo"]
        if cache:
            self._cache = (c, h)
        return q

    def _backward(self, dq: np.ndarray) -> Dict[str, np.ndarray]:
        c, h_last = self._cache
        g: Dict[str, np.ndarray] = {}
        B = dq.shape[0]
        g["Wo"] = np.einsum("bif,bia->fa", h_last, dq)
        g["bo"] = dq.sum(axis=(0, 1))
        dh = dq @ self.p["Wo"].T
        for r in reversed(range(self.R)):
            h_in, m, h_out = c[r]
            dz = dh * (1.0 - h_out ** 2)
            g[f"Ws{r}"] = np.einsum("bif,bih->fh", h_in, dz)
            g[f"Wn{r}"] = np.einsum("bif,bih->fh", m, dz)
            g[f"b{r}"] = dz.sum(axis=(0, 1))
            dm = dz @ self.p[f"Wn{r}"].T
            dh = dz @ self.p[f"Ws{r}"].T + np.einsum("ji,bjf->bif", self.A, dm)
        return g

    def act(self, obs, state, eps):
        q = self._forward(obs[None, :, :], self.p, cache=False)[0]
        acts = []
        for i in range(self.n_agents):
            if self.rng.random() < eps:
                acts.append(int(self.rng.integers(self.n_actions)))
            else:
                acts.append(int(np.argmax(q[i])))
        return acts

    def observe(self, obs, state, actions, reward, obs2, state2, done):
        self.buf.add(obs, state, actions, reward, obs2, state2, done)

    def train_step(self):
        tr = self.tr
        if self.buf.n < tr.warmup_transitions:
            return
        self.steps += 1
        if self.steps % tr.train_every:
            return
        o, s, a, r, o2, s2, d = self.buf.sample(tr.batch_size)
        B = len(r)
        qn = self._forward(o2, self.tp, cache=False).max(axis=2)      # (B, n_agents)
        # independent Q-learning with parameter sharing: each node bootstraps on the
        # shared global reward divided equally between the cells
        target = (r / self.n_agents)[:, None] + tr.gamma * (1.0 - d)[:, None] * qn
        q = self._forward(o, self.p, cache=True)
        idx_b = np.arange(B)[:, None].repeat(self.n_agents, 1)
        idx_i = np.arange(self.n_agents)[None, :].repeat(B, 0)
        pred = q[idx_b, idx_i, a]
        dq = np.zeros_like(q)
        dq[idx_b, idx_i, a] = huber_grad(pred - target) / (B * self.n_agents)
        g = self._backward(dq)
        self.opt.step(self.p, g, tr.grad_clip)
        if self.steps % tr.target_update_steps == 0:
            self.tp = {k: v.copy() for k, v in self.p.items()}


AGENT_REGISTRY = {
    "DQN": DQNAgent,
    "MADRL-QMIX": QMIXAgent,
    "FedAvg-DQN": FedAvgAgent,
    "GNN-DRL": GNNDQNAgent,
}
ALGORITHMS = tuple(AGENT_REGISTRY.keys())


# ==================================================================================
# 4. Training loop and convergence diagnostics
# ==================================================================================

def epsilon(ep: int, tr: RLTrainConfig) -> float:
    frac = min(1.0, ep / max(tr.eps_decay_episodes, 1))
    return tr.eps_start + frac * (tr.eps_end - tr.eps_start)


def random_policy_return(env_seed: int, n_episodes: int,
                         env_cfg: RLEnvConfig = RLENV) -> float:
    """Mean episode return of a uniformly random policy: the 'no learning' floor."""
    rng = np.random.default_rng(env_seed)
    env = MultiCellPowerEnv(rng, env_cfg)
    tot = []
    for _ in range(n_episodes):
        env.reset()
        acc = 0.0
        done = False
        while not done:
            a = rng.integers(env.n_levels, size=env.n_cells)
            _, _, r, done, _ = env.step(a)
            acc += r
        tot.append(acc)
    return float(np.mean(tot))


class RunningNorm:
    """Welford running mean/std used to whiten the reward stream.

    The environment reward has a large positive offset (the sum-log-rate utility of
    20 users), so raw action-value differences are ~0.3 % of the action value itself.
    Whitening the reward is a standard and label-preserving transformation: it does
    not change the ordering of policies, only the conditioning of the regression the
    Q-network has to solve.  Evaluation always uses the *raw* reward.
    """

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(np.sqrt(self.m2 / (self.n - 1)), 1e-6)

    def normalise(self, x: float) -> float:
        return (x - self.mean) / self.std


def evaluate(agent: BaseAgent, eval_seed: int, n_episodes: int,
             env_cfg: RLEnvConfig = RLENV,
             with_optimum: bool = False) -> Tuple[float, float]:
    """Greedy-policy return on a FIXED set of evaluation scenarios.

    The environment is rebuilt from `eval_seed` at every call, so every evaluation
    during training replays exactly the same channel realisations and arrivals.  This
    removes scenario noise from the learning curve; what is left is genuine policy
    variation.
    """
    tot, opt = [], []
    for e in range(n_episodes):
        env = MultiCellPowerEnv(np.random.default_rng(eval_seed + 1000 * e), env_cfg)
        obs, state = env.reset()
        acc, acc_opt = 0.0, 0.0
        done = False
        while not done:
            if with_optimum:
                acc_opt += env.myopic_optimum()
            a = agent.act(obs, state, 0.0)
            obs, state, r, done, _ = env.step(a)
            acc += r
        tot.append(acc)
        opt.append(acc_opt)
    return float(np.mean(tot)), (float(np.mean(opt)) if with_optimum else float("nan"))


def fixed_scenario_floor(eval_seed: int, n_episodes: int,
                         env_cfg: RLEnvConfig = RLENV) -> Tuple[float, float]:
    """Random-policy and all-max-power returns on the same fixed evaluation scenarios."""
    rand, maxp = [], []
    for e in range(n_episodes):
        env = MultiCellPowerEnv(np.random.default_rng(eval_seed + 1000 * e), env_cfg)
        env.reset()
        rng = np.random.default_rng(eval_seed + 7)
        acc = 0.0
        done = False
        while not done:
            _, _, r, done, _ = env.step(rng.integers(env.n_levels, size=env.n_cells))
            acc += r
        rand.append(acc)
        env = MultiCellPowerEnv(np.random.default_rng(eval_seed + 1000 * e), env_cfg)
        env.reset()
        acc = 0.0
        done = False
        while not done:
            _, _, r, done, _ = env.step([env.n_levels - 1] * env.n_cells)
            acc += r
        maxp.append(acc)
    return float(np.mean(rand)), float(np.mean(maxp))


def convergence_diagnostics(curve: np.ndarray, floor: float,
                            tr: RLTrainConfig = RLTRAIN) -> dict:
    """Decide, from the data, whether a run converged.

    Criteria (all three must hold):
      1. improvement: final plateau mean > random-policy floor by more than one
         standard deviation of the plateau;
      2. stability: |relative slope| over the last 20 % of training below
         `convergence_tolerance`;
      3. monotone progress: plateau mean > mean of the first 20 % of training.
    """
    n = len(curve)
    k = max(3, n // 5)
    tail = curve[-k:]
    head = curve[:k]
    x = np.arange(k, dtype=float)
    slope = float(np.polyfit(x, tail, 1)[0])
    scale = max(abs(float(tail.mean())), 1e-9)
    rel_slope = slope * k / scale
    sd = float(tail.std(ddof=1)) if k > 1 else 0.0
    improved = bool(tail.mean() > floor + sd)
    stable = bool(abs(rel_slope) < tr.convergence_tolerance)
    progressed = bool(tail.mean() > head.mean())
    # first episode at which the moving average reaches 95 % of the plateau level
    w = min(tr.convergence_window, max(2, n // 10))
    ma = np.convolve(curve, np.ones(w) / w, mode="valid")
    thresh = floor + 0.95 * (tail.mean() - floor)
    reach = np.nonzero(ma >= thresh)[0]
    ep95 = int(reach[0] + w) if reach.size else -1
    return {
        "plateau_return": float(tail.mean()),
        "plateau_sd": sd,
        "relative_slope": rel_slope,
        "improved_over_random": improved,
        "stable": stable,
        "progressed": progressed,
        "converged": bool(improved and stable and progressed),
        "episodes_to_95pct": ep95,
    }


EVAL_SEED_OFFSET = 900_000
N_EVAL_SCENARIOS = 8


def train_agent(algorithm: str, seed: int, n_episodes: int | None = None,
                env_cfg: RLEnvConfig = RLENV, tr: RLTrainConfig = RLTRAIN,
                eval_every: int = 20) -> dict:
    """Train one algorithm for one seed.  Returns the learning curve plus diagnostics.

    Train / test separation: training scenarios are drawn from `seed`, evaluation
    scenarios from a disjoint seed stream (`EVAL_SEED_OFFSET + seed`) and are held
    fixed for the whole run, so the learning curve measures policy improvement on
    scenarios the agent never trained on.
    """
    n_episodes = tr.n_episodes if n_episodes is None else n_episodes
    rng = np.random.default_rng(seed)
    env = MultiCellPowerEnv(np.random.default_rng(seed + 1), env_cfg)
    eval_seed = EVAL_SEED_OFFSET + seed
    agent = AGENT_REGISTRY[algorithm](rng, env, tr)
    norm = RunningNorm()

    train_curve = np.zeros(n_episodes)
    eval_eps, eval_ret, eval_opt = [], [], []
    for ep in range(n_episodes):
        obs, state = env.reset()
        eps = epsilon(ep, tr)
        acc = 0.0
        done = False
        while not done:
            a = agent.act(obs, state, eps)
            obs2, state2, r, done, _ = env.step(a)
            norm.update(r)
            agent.observe(obs, state, a, norm.normalise(r), obs2, state2, done)
            agent.train_step()
            obs, state = obs2, state2
            acc += r
        agent.end_episode(ep)
        train_curve[ep] = acc
        if (ep + 1) % eval_every == 0 or ep == 0:
            ret, _ = evaluate(agent, eval_seed, N_EVAL_SCENARIOS, env_cfg,
                              with_optimum=False)
            eval_eps.append(ep + 1)
            eval_ret.append(ret)

    floor, maxp = fixed_scenario_floor(eval_seed, N_EVAL_SCENARIOS, env_cfg)
    diag = convergence_diagnostics(np.asarray(eval_ret), floor, tr)
    final_ret, final_opt = evaluate(agent, eval_seed, N_EVAL_SCENARIOS, env_cfg,
                                    with_optimum=True)
    denom = max(final_opt - floor, 1e-9)
    curve_score = [(v - floor) / denom for v in eval_ret]
    return {
        "algorithm": algorithm,
        "seed": seed,
        "train_curve": [float(v) for v in train_curve],
        "eval_episodes": eval_eps,
        "eval_return": [float(v) for v in eval_ret],
        "normalised_score_curve": [float(v) for v in curve_score],
        "random_floor": floor,
        "max_power_reference": maxp,
        "final_return": final_ret,
        "final_myopic_optimum": final_opt,
        # Normalised score: 0 = uniformly random power control,
        #                   1 = per-slot myopic-greedy oracle with full information.
        "normalised_score": float((final_ret - floor) / denom),
        "final_pct_of_myopic_optimum": 100.0 * final_ret / final_opt if final_opt else float("nan"),
        "gain_over_max_power_pct": 100.0 * (final_ret - maxp) / abs(maxp),
        "diagnostics": diag,
        "n_episodes": n_episodes,
        "extra": {"fedavg_rounds": getattr(agent, "n_rounds", None)},
    }
