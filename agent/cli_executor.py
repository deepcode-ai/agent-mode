"""
CLI Executor module for securely executing shell commands.

This module provides a CLIExecutor class that handles secure command execution
with proper permission checks, validation, sanitization, and logging.
"""

import logging
import os
import re
import shlex
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "cli_executor.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("cli_executor")


class CommandVerdict(Enum):
    """Enumeration of possible command validation results."""

    ALLOWED = "allowed"
    DENIED = "denied"
    NEEDS_APPROVAL = "needs_approval"


class CLIExecutor:
    """
    A secure command line executor that handles permission checks,
    validation, and sanitization of shell commands.
    """

    # Set of potentially dangerous commands that require extra scrutiny
    DANGEROUS_COMMANDS = {
        "rm", "dd", "mkfs", "format", "fdisk", "shutdown", "reboot",
        "halt", "poweroff", ">", ">>", "mv", "chmod", "chown"
    }

    # Commands that are completely blocked
    BLOCKED_COMMANDS = {
        "sudo", "su", "doas", "pkexec"
    }

    def __init__(self, permission_manager=None, context_manager=None):
        """
        Initialize the CLI Executor.

        Args:
            permission_manager: Component responsible for permission decisions
            context_manager: Component that tracks environment and state
        """
        self.permission_manager = permission_manager
        self.context_manager = context_manager
        self.command_history = []
        logger.info("CLIExecutor initialized")

    def execute(
        self, 
        command: str, 
        check_permissions: bool = True, 
        capture_output: bool = True,
        working_dir: Optional[str] = None, 
        env_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = 60
    ) -> Dict[str, Union[str, int]]:
        """
        Execute a shell command with security checks.

        Args:
            command: The command to execute
            check_permissions: Whether to check permissions before execution
            capture_output: Whether to capture and return command output
            working_dir: Working directory for command execution
            env_vars: Additional environment variables for the command
            timeout: Command timeout in seconds

        Returns:
            Dict containing output, error, and exit code
        """
        # Always log the command request
        logger.info(f"Command execution requested: {command}")
        
        # Validate and sanitize the command
        sanitized_command = self._sanitize_command(command)
        if not sanitized_command:
            return {
                "output": "", 
                "error": "Command was empty after sanitization", 
                "exit_code": 1
            }
        
        # Check permissions if required
        if check_permissions:
            verdict, reason = self._check_permissions(sanitized_command)
            if verdict != CommandVerdict.ALLOWED:
                logger.warning(f"Command denied: {sanitized_command}. Reason: {reason}")
                return {
                    "output": "", 
                    "error": f"Permission denied: {reason}", 
                    "exit_code": 1
                }
        
        # Prepare environment
        execution_env = os.environ.copy()
        if env_vars:
            execution_env.update(env_vars)
        
        # Execute the command
        try:
            logger.info(f"Executing command: {sanitized_command}")
            result = self._run_command(
                sanitized_command, 
                capture_output=capture_output,
                cwd=working_dir,
                env=execution_env,
                timeout=timeout
            )
            
            # Add to command history
            self.command_history.append({
                "command": sanitized_command,
                "exit_code": result["exit_code"],
                "timestamp": self.context_manager.get_timestamp() if self.context_manager else None
            })
            
            # Log the result
            if result["exit_code"] == 0:
                logger.info(f"Command executed successfully: {sanitized_command}")
            else:
                logger.error(
                    f"Command failed with exit code {result['exit_code']}: {sanitized_command}"
                )
                
            return result
            
        except Exception as e:
            logger.exception(f"Error executing command: {sanitized_command}")
            return {"output": "", "error": str(e), "exit_code": 1}

    def _sanitize_command(self, command: str) -> str:
        """
        Sanitize a command by removing potentially dangerous elements.

        Args:
            command: The command to sanitize

        Returns:
            The sanitized command string
        """
        # Trim whitespace
        command = command.strip()
        
        # Check for empty command
        if not command:
            return ""
        
        # Basic sanitization to prevent command injection
        # Remove backticks and $(command) syntax
        command = re.sub(r'`.*?`', '', command)
        command = re.sub(r'\$\(.*?\)', '', command)
        
        # Check for pipe chains to other dangerous commands
        for dangerous_cmd in self.BLOCKED_COMMANDS:
            if re.search(fr'[|;]\s*{dangerous_cmd}\b', command):
                logger.warning(f"Blocked dangerous command in pipe chain: {dangerous_cmd}")
                return ""
        
        return command

    def _check_permissions(self, command: str) -> Tuple[CommandVerdict, str]:
        """
        Check if the command is allowed to be executed.

        Args:
            command: The command to check

        Returns:
            Tuple of (verdict, reason)
        """
        # Parse the command to get the base executable
        try:
            cmd_parts = shlex.split(command)
            base_cmd = cmd_parts[0] if cmd_parts else ""
        except ValueError:
            return CommandVerdict.DENIED, "Invalid command syntax"
        
        # Check for blocked commands
        if base_cmd in self.BLOCKED_COMMANDS:
            return CommandVerdict.DENIED, f"Command '{base_cmd}' is blocked for security reasons"
        
        # Check for dangerous commands
        if base_cmd in self.DANGEROUS_COMMANDS:
            if self.permission_manager:
                # Delegate to permission manager if available
                if self.permission_manager.check_permission(command):
                    return CommandVerdict.ALLOWED, "Approved by permission manager"
                else:
                    return CommandVerdict.NEEDS_APPROVAL, f"'{base_cmd}' requires explicit approval"
            else:
                return CommandVerdict.NEEDS_APPROVAL, f"'{base_cmd}' is potentially dangerous"
        
        # Use permission manager if available
        if self.permission_manager:
            if self.permission_manager.check_permission(command):
                return CommandVerdict.ALLOWED, "Approved by permission manager"
            else:
                return CommandVerdict.NEEDS_APPROVAL, "Not in allowed commands list"
        
        # Default allow if no permission manager is available
        # In a real implementation, this might need to be more restrictive
        return CommandVerdict.ALLOWED, "Default allow policy"

    def _run_command(
        self, 
        command: str, 
        capture_output: bool = True, 
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Union[str, int]]:
        """
        Run a shell command and return its output.

        Args:
            command: The command to execute
            capture_output: Whether to capture the command output
            cwd: Working directory
            env: Environment variables
            timeout: Command timeout in seconds

        Returns:
            Dict containing output, error, and exit code
        """
        try:
            # Execute the command
            process = subprocess.run(
                command,
                shell=True,  # Use shell for command execution
                text=True,  # Return strings rather than bytes
                capture_output=capture_output,
                cwd=cwd,
                env=env,
                timeout=timeout
            )
            
            return {
                "output": process.stdout if capture_output else "",
                "error": process.stderr if capture_output else "",
                "exit_code": process.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "exit_code": 124  # Standard timeout exit code
            }
        except Exception as e:
            return {
                "output": "",
                "error": str(e),
                "exit_code": 1
            }

    def validate_command_syntax(self, command: str) -> bool:
        """
        Validate that the command has valid shell syntax.

        Args:
            command: Command to validate

        Returns:
            True if the syntax is valid, False otherwise
        """
        try:
            # Try to parse the command using shlex to check syntax
            shlex.split(command)
            return True
        except ValueError:
            return False

    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        Get the recent command execution history.

        Args:
            limit: Maximum number of history items to return

        Returns:
            List of command history entries
        """
        return self.command_history[-limit:] if self.command_history else []

