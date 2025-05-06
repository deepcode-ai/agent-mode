# agent/permissions.py
class PermissionManager:
    def __init__(self):
        self.permissions = {}
    
    def add_permission(self, plugin, command):
        self.permissions[plugin] = command
    
    def check_permission(self, plugin, command):
        return self.permissions.get(plugin) == command
