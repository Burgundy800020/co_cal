import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cocal.conformal import conformal_quantile
from cocal.replay_buffer import ReplayBuffer
from cocal.td3_bc import TD3BC


def test_conformal_quantile():
    assert conformal_quantile(np.arange(100), 0.1) >= 89


def test_td3bc_update():
    buf = ReplayBuffer(3, 2, 500)
    buf.add_batch(np.random.randn(500, 3), np.random.uniform(-1, 1, (500, 2)), np.random.randn(500),
                  np.random.randn(500, 3), np.zeros(500))
    agent = TD3BC(3, 2, 1.0)
    for _ in range(4):
        s = agent.update(buf, 32)
    assert "actor_loss" in s
