from __future__ import annotations

from types import SimpleNamespace

import diffulex.config as config_module
from diffulex.config import Config


def test_standard_moe_backend_ignores_requested_ep_size(monkeypatch):
    monkeypatch.setattr(config_module.os.path, "isdir", lambda _path: True)
    monkeypatch.setattr(
        config_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(max_position_embeddings=2048),
    )

    config = Config(
        model="/fake/model",
        tensor_parallel_size=2,
        expert_parallel_size=8,
        data_parallel_size=1,
        moe_dispatcher_backend="standard",
        device_ids=[0, 1],
    )

    assert config.expert_parallel_size == 1
    assert config.tensor_parallel_size == 2
