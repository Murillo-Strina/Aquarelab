import pytest

from aquarelab.train import load_config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[run]\nname = "base"\n\n[loss]\nentropy_weight = 0.01\n')
    return path


def test_overrides_parse_numbers_and_strings(config_file):
    cfg = load_config(config_file, ["loss.entropy_weight=0.05", "run.name=entropy_005"])
    assert cfg["loss"]["entropy_weight"] == 0.05
    assert cfg["run"]["name"] == "entropy_005"


def test_unknown_key_is_rejected(config_file):
    with pytest.raises(KeyError):
        load_config(config_file, ["loss.entropy_wieght=0.05"])  # typo on purpose
