import pytest

from agent_traces.config import Config, ModelConfig, load_config


def test_missing_env_is_actionable(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[model]\nname="${NONEXISTENT_ATC_MODEL_ENV}"\n')
    with pytest.raises(ValueError, match="Unresolved environment"):
        load_config(p)


@pytest.mark.parametrize("override", [{"n": 2}, {"stream": True}, {"max_tokens": 1}])
def test_extra_body_cannot_override_accounting_invariants(override):
    with pytest.raises(ValueError, match="controlled request field"):
        Config(model=ModelConfig(name="fixture", extra_body=override)).validate()


@pytest.mark.parametrize("url", ["http://user:key@host/v1", "http://host/v1?api_key=secret"])
def test_credentials_do_not_go_in_logged_urls(url):
    with pytest.raises(ValueError, match="credentials"):
        Config(model=ModelConfig(name="fixture", base_url=url)).validate()
