# agent/permission_manager.py
import yaml

class PermissionManager:
    def __init__(self, config_path="config/permissions.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def is_allowed(self, plugin_name: str, user_role: str = "default") -> bool:
        allowed_plugins = self.config.get("roles", {}).get(user_role, [])
        return plugin_name in allowed_plugins or "*" in allowed_plugins
