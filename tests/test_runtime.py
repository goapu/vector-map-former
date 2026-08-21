from pathlib import Path

import torch

from vector_map_former.runtime import atomic_torch_save, runtime_metadata


def test_checkpoint_metadata_is_weights_only_safe(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    payload = {"state": {"weight": torch.ones(2)}, "runtime": runtime_metadata()}
    atomic_torch_save(payload, path)
    restored = torch.load(path, weights_only=True)
    assert restored["runtime"]["torch"] == str(torch.__version__)
