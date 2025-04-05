# tests/test_plugins.py
from agent.plugin_loader import PluginLoader

def test_plugin_loading():
    loader = PluginLoader()
    plugins = loader.load_plugins()
    assert len(plugins) > 0

