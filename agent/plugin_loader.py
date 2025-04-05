# agent/plugin_loader.py
import importlib
import yaml
from pathlib import Path

class PluginLoader:
    def __init__(self, config_path="config/plugins.yaml"):
        self.plugins = []
        self.config_path = config_path

    def load_plugins(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            plugin_names = config.get("plugins", [])

        for plugin_name in plugin_names:
            try:
                module = importlib.import_module(f"plugins.{plugin_name}")
                plugin_class = getattr(module, f"{plugin_name.capitalize()}Plugin")
                self.plugins.append(plugin_class())
            except Exception as e:
                print(f"Failed to load plugin '{plugin_name}': {e}")

        return self.plugins

