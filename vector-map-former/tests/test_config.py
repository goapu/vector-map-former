from pathlib import Path

import pytest

from vector_map_former.config import load_config, project_config_from_dict


def _valid_config() -> dict[str, object]:
    return {
        "seed": 1,
        "data": {},
        "model": {},
        "training": {},
        "output": {},
    }


def test_load_baseline_config() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "baseline.yaml"
    config = load_config(config_path)
    assert config.model.name == "mlp"
    assert config.data.max_vertices == 32


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
seed: 1
data: {data_dir: data, unexpected: true}
model: {}
training: {}
output: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys in 'data'"):
        load_config(path)


def test_config_section_must_be_mapping() -> None:
    raw = _valid_config()
    raw["model"] = []
    with pytest.raises(ValueError, match="must be a mapping"):
        project_config_from_dict(raw)


def test_mlp_hidden_dimension_is_not_constrained_by_attention_heads() -> None:
    raw = _valid_config()
    raw["model"] = {"name": "mlp", "hidden_dim": 95}
    project_config_from_dict(raw)


def test_transformer_hidden_dimension_must_match_attention_heads() -> None:
    raw = _valid_config()
    raw["model"] = {"name": "ring_transformer", "hidden_dim": 95}
    with pytest.raises(ValueError, match="divisible"):
        project_config_from_dict(raw)
