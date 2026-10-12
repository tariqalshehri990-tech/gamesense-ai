import pytest
from nafas.agents import build_facts

@pytest.fixture(scope="session")
def facts():
    return build_facts(35)
