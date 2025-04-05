"""
Permission management module for handling command execution approvals.

This module provides a PermissionManager class that handles permission checks,
approval workflows, and configuration loading from YAML files.
"""

import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "permission.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("permission")


class PermissionLevel(Enum):
    """Enumeration of permission levels for operations."""

    ALLOW = "allow"  # Always allowed without prompting
    PROMPT = "prompt"  # Requires user confirmation
    DENY = "deny"  # Always denied


class PermissionCategory(Enum):
    """Categories of permissions that can be granted."""

    COMMAND = "command"  # Shell command execution
    PLUGIN = "plugin"  # Plugin operations
    FILE_READ = "file_read"  # File reading operations
    FILE_WRITE = "file_write"  # File writing operations
    NETWORK = "network"  # Network access
    SYSTEM = "system"  # System modifications


class ApprovalState(Enum):
    """Possible states for an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class PermissionManager:
    """
    Manages permissions for agent operations, handling configuration loading,
    permission checks, and user approval workflows.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        approval_timeout: int = 300,
        approval_callback=None,
    ):
        """
        Initialize the Permission Manager.

        Args:
            config_path: Path to the permissions YAML config file
            approval_timeout: Timeout in seconds for approval requests
            approval_callback: Function to call when approval is needed
        """
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "permissions.yaml"
        self.approval_timeout = approval_timeout
        self.approval_callback = approval_callback
        
        # Permission configuration
        self.permissions: Dict[str, Dict[str, Any]] = {}
        
        # Permission cache for performance
        self.permission_cache: Dict[str, Tuple[bool, float]] = {}
        
        # Approval tracking
        self.pending_approvals: Dict[str, Dict] = {}
        self.approved_items: Set[str] = set()
        
        # Load permissions from config
        self._load_permissions()
        
        logger.info("PermissionManager initialized")

    def _load_permissions(self) -> None:
        """Load permissions from the YAML configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.permissions = yaml.safe_load(f) or {}
                logger.info(f"Loaded permissions from {self.config_path}")
            else:
                logger.warning(f"Permissions file not found at {self.config_path}, using default permissions")
                self.permissions = self._default_permissions()
        except Exception as e:
            logger.error(f"Error loading permissions: {str(e)}")
            self.permissions = self._default_permissions()
    
    def _default_permissions(self) -> Dict[str, Dict[str, Any]]:
        """Create default permission configuration."""
        return {
            "commands": {
                "default": PermissionLevel.PROMPT.value,
                "allowed": [
                    "ls", "cat", "echo", "cd", "pwd", "mkdir", "cp", "mv", 
                    "touch", "grep", "find", "ps", "git", "npm", "node"
                ],
                "prompt": [
                    "rm", "kill", "chmod", "chown"
                ],
                "denied": [
                    "sudo", "su", "doas", "pkexec"
                ]
            },
            "plugins": {
                "default": PermissionLevel.PROMPT.value,
                "allowed": ["github", "aws_readonly", "kubernetes_readonly"],
                "prompt": ["aws", "kubernetes"],
                "denied": []
            },
            "files": {
                "read": {
                    "default": PermissionLevel.ALLOW.value,
                    "denied_patterns": [
                        ".*\\.env", ".*secret.*", ".*password.*", 
                        ".*credential.*", ".*key.*\\.pem"
                    ]
                },
                "write": {
                    "default": PermissionLevel.PROMPT.value,
                    "allowed_patterns": [
                        ".*\\.txt", ".*\\.md", ".*\\.json", ".*\\.yaml", 
                        ".*\\.py", ".*\\.js", ".*\\.ts", ".*\\.html", ".*\\.css"
                    ],
                    "denied_patterns": [
                        ".*\\.env", "/etc/.*", "/var/.*", "/usr/.*",
                        "/bin/.*", "/sbin/.*"
                    ]
                }
            },
            "network": {
                "default": PermissionLevel.PROMPT.value,
                "allowed_domains": [
                    "github.com", "gitlab.com", "bitbucket.org",
                    "npmjs.com", "pypi.org"
                ]
            }
        }

    def save_permissions(self) -> bool:
        """Save current permissions to the configuration file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, "w") as f:
                yaml.dump(self.permissions, f, default_flow_style=False)
            
            logger.info(f"Saved permissions to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving permissions: {str(e)}")
            return False

    def check_permission(
        self, 
        action: str, 
        category: Union[PermissionCategory, str] = PermissionCategory.COMMAND,
        context: Optional[Dict[str, Any]] = None,
        request_approval: bool = True
    ) -> bool:
        """
        Check if an action is permitted.

        Args:
            action: The action to check (command, plugin name, etc.)
            category: The category of the permission
            context: Additional context for the permission check
            request_approval: Whether to request approval if needed

        Returns:
            True if the action is permitted, False otherwise
        """
        if isinstance(category, PermissionCategory):
            category = category.value
            
        # Check cache first for performance
        cache_key = f"{category}:{action}"
        if cache_key in self.permission_cache:
            is_allowed, timestamp = self.permission_cache[cache_key]
            
            # Cache entries expire after 1 hour
            if time.time() - timestamp < 3600:
                return is_allowed
                
        # Check if previously approved
        if cache_key in self.approved_items:
            return True
            
        # Parse command if it's a shell command
        if category == PermissionCategory.COMMAND.value:
            command_parts = action.split()
            base_command = command_parts[0] if command_parts else ""
            
            # Check specific command permissions
            return self._check_command_permission(base_command, action, request_approval)
            
        elif category == PermissionCategory.PLUGIN.value:
            # Check plugin permissions
            return self._check_plugin_permission(action, request_approval)
            
        elif category == PermissionCategory.FILE_READ.value:
            # Check file read permissions
            return self._check_file_permission(action, "read", request_approval)
            
        elif category == PermissionCategory.FILE_WRITE.value:
            # Check file write permissions
            return self._check_file_permission(action, "write", request_approval)
            
        elif category == PermissionCategory.NETWORK.value:
            # Check network permissions
            return self._check_network_permission(action, request_approval)
            
        else:
            # Unknown category, default to requiring approval
            logger.warning(f"Unknown permission category: {category}")
            if request_approval:
                return self._request_approval(f"Unknown operation type: {category} - {action}")
            return False

    def _check_command_permission(self, base_command: str, full_command: str, request_approval: bool) -> bool:
        """Check permissions for a shell command."""
        commands_config = self.permissions.get("commands", {})
        default_level = commands_config.get("default", PermissionLevel.PROMPT.value)
        
        # Check if command is in allowed list
        if base_command in commands_config.get("allowed", []):
            self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", True)
            return True
            
        # Check if command is in denied list
        if base_command in commands_config.get("denied", []):
            logger.warning(f"Command denied by configuration: {base_command}")
            self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", False)
            return False
            
        # Check if command requires prompting
        if base_command in commands_config.get("prompt", []):
            if request_approval:
                approved = self._request_approval(
                    f"Execute command: {full_command}", 
                    f"This command ({base_command}) requires explicit approval."
                )
                self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", approved)
                if approved:
                    self.approved_items.add(f"{PermissionCategory.COMMAND.value}:{full_command}")
                return approved
            return False
            
        # Handle based on default level
        if default_level == PermissionLevel.ALLOW.value:
            self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", True)
            return True
        elif default_level == PermissionLevel.DENY.value:
            self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", False)
            return False
        else:  # PROMPT
            if request_approval:
                approved = self._request_approval(f"Execute command: {full_command}")
                self._update_cache(f"{PermissionCategory.COMMAND.value}:{full_command}", approved)
                if approved:
                    self.approved_items.add(f"{PermissionCategory.COMMAND.value}:{full_command}")
                return approved
            return False

    def _check_plugin_permission(self, plugin_name: str, request_approval: bool) -> bool:
        """Check permissions for a plugin operation."""
        plugins_config = self.permissions.get("plugins", {})
        default_level = plugins_config.get("default", PermissionLevel.PROMPT.value)
        
        # Similar logic to command permission checks
        if plugin_name in plugins_config.get("allowed", []):
            return True
        if plugin_name in plugins_config.get("denied", []):
            return False
        if plugin_name in plugins_config.get("prompt", []) or default_level == PermissionLevel.PROMPT.value:
            if request_approval:
                return self._request_approval(f"Use plugin: {plugin_name}")
            return False
        
        return default_level == PermissionLevel.ALLOW.value

    def _check_file_permission(self, file_path: str, operation: str, request_approval: bool) -> bool:
        """Check permissions for file operations."""
        files_config = self.permissions.get("files", {}).get(operation, {})
        default_level = files_config.get("default", PermissionLevel.PROMPT.value)
        
        # Check denied patterns first
        for pattern in files_config.get("denied_patterns", []):
            if re.match(pattern, file_path):
                logger.warning(f"File operation denied by pattern match: {pattern} - {file_path}")
                return False
        
        # Check allowed patterns
        for pattern in files_config.get("allowed_patterns", []):
            if re.match(pattern, file_path):
                return True
        
        # Handle based on default level
        if default_level == PermissionLevel.ALLOW.value:
            return True
        elif default_level == PermissionLevel.DENY.value:
            return False
        else:  # PROMPT
            if request_approval:
                return self._request_approval(f"{operation.capitalize()} file: {file_path}")
            return False

    def _check_network_permission(self, domain: str, request_approval: bool) -> bool:
        """Check permissions for network operations."""
        network_config = self.permissions.get("network", {})
        default_level = network_config.get("default", PermissionLevel.PROMPT.value)
        
        # Check if domain is in allowed list
        for allowed_domain in network_config.get("allowed_domains", []):
            if domain.endswith(allowed_domain):
                return True
        
        # Handle based on default level
        if default_level == PermissionLevel.ALLOW.value:
            return True
        elif default_level == PermissionLevel.DENY.value:
            return False
        else:  # PROMPT
            if request_approval:
                return self._request_approval(f"Network access to: {domain}")
            return False

    def _request_approval(self, action: str, reason: Optional[str] = None) -> bool:
        """
        Request approval for an action.

        Args:
            action: The action requiring approval
            reason: The reason approval is needed

        Returns:
            True if approved, False otherwise
        """
        # Generate a unique ID for this approval request
        approval_id = f"approval_{int(time.time())}_{hash(action) % 10000}"
        
        # Create the approval request
        approval_request = {
            "id": approval_id,
            "action": action,
            "reason": reason,
            "timestamp": time.time(),
            "state": ApprovalState.PENDING.value,
        }
        
        # Store the approval request
        self.pending_approvals[approval_id] = approval_request
        
        # If a callback is registered, use it to request approval
        if self.approval_callback:
            try:
                result = self.approval_callback(approval_request)
                self._update_approval_state(approval_id, 
                                          ApprovalState.APPROVED if result else ApprovalState.DENIED)
                return result
            except Exception as e:
                logger.error(f"Error in approval callback: {str(e)}")
                self._update_approval_state(approval_id, ApprovalState.DENIED)
                return False
        
        # Default implementation if no callback is provided - console prompt
        prompt_text = f"Permission required: {action}"
        if reason:
            prompt_text += f"\nReason: {reason}"
        prompt_text += "\nAllow

