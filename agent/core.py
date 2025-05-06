import os
from dotenv import load_dotenv

load_dotenv()

class RoleManager:
    def __init__(self, role: str = None):
        self.role = role or os.getenv("AGENT_ROLE", "user")

    def has_permission(self, action: str) -> bool:
        # Add your role-action logic here
        if self.role == "admin":
            return True
        if self.role == "developer" and action in ["read", "write"]:
            return True
        return action == "read"
