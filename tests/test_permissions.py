# tests/test_permissions.py
from agent.permission_manager import PermissionManager

def test_permissions():
    pm = PermissionManager()
    assert pm.is_allowed("github", "default")
    assert not pm.is_allowed("aws", "default")
    assert pm.is_allowed("aws", "admin")
