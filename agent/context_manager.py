"""
Context management module for tracking execution state.

This module provides a ContextManager class that handles environment variables,
command history, working directory tracking, and session state persistence.
"""

import datetime
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "context_manager.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("context_manager")


class ContextManager:
    """
    Maintains execution context for the agent, including environment variables,
    command history, working directory, and session state.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        context_file: Optional[str] = None,
        max_history_size: int = 100,
    ):
        """
        Initialize the Context Manager.

        Args:
            session_id: Unique identifier for this session
            context_file: Path to the context file for persistence
            max_history_size: Maximum number of history entries to keep
        """
        self.session_id = session_id or f"session_{int(time.time())}"
        self.context_file = context_file or Path(__file__).parent.parent / "data" / "user_context.json"
        self.max_history_size = max_history_size
        
        # Current state
        self.current_dir = os.getcwd()
        self.env_vars = dict(os.environ)
        
        # History tracking
        self.command_history: List[Dict[str, Any]] = []
        self.output_history: List[Dict[str, Any]] = []
        
        # Session metadata
        self.session_start_time = datetime.datetime.now().isoformat()
        self.system_info = self._get_system_info()
        
        # User context
        self.user_variables: Dict[str, Any] = {}
        self.preferences: Dict[str, Any] = {}
        
        # Plugins context
        self.plugin_states: Dict[str, Dict[str, Any]] = {}
        
        # Load existing context if available
        self._load_context()
        
        logger.info(f"ContextManager initialized with session ID: {self.session_id}")

    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        system_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
        }
        
        # Get shell information
        try:
            shell = os.environ.get("SHELL", "")
            if shell:
                # Get shell version
                try:
                    result = subprocess.run(
                        f"{shell} --version", 
                        shell=True, 
                        capture_output=True, 
                        text=True
                    )
                    if result.returncode == 0:
                        system_info["shell_version"] = result.stdout.strip()
                except Exception:
                    pass
                system_info["shell"] = os.path.basename(shell)
        except Exception as e:
            logger.warning(f"Failed to get shell information: {str(e)}")
            
        return system_info

    def _load_context(self) -> None:
        """Load context from the context file if it exists."""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, "r") as f:
                    context_data = json.load(f)
                
                # Restore relevant parts of the context
                self.user_variables = context_data.get("user_variables", {})
                self.preferences = context_data.get("preferences", {})
                self.plugin_states = context_data.get("plugin_states", {})
                
                # Only load command history if session_id matches
                if context_data.get("session_id") == self.session_id:
                    self.command_history = context_data.get("command_history", [])
                    self.output_history = context_data.get("output_history", [])
                
                logger.info(f"Loaded context from {self.context_file}")
        except Exception as e:
            logger.error(f"Error loading context: {str(e)}")

    def save_context(self) -> bool:
        """Save the current context to the context file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.context_file), exist_ok=True)
            
            # Prepare context data
            context_data = {
                "session_id": self.session_id,
                "session_start_time": self.session_start_time,
                "last_updated": datetime.datetime.now().isoformat(),
                "user_variables": self.user_variables,
                "preferences": self.preferences,
                "plugin_states": self.plugin_states,
                "command_history": self.command_history[-self.max_history_size:],
                "output_history": self.output_history[-self.max_history_size:],
                "system_info": self.system_info,
                "current_dir": self.current_dir,
            }
            
            # Write to file
            with open(self.context_file, "w") as f:
                json.dump(context_data, f, indent=2)
            
            logger.info(f"Saved context to {self.context_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving context: {str(e)}")
            return False

    def get_timestamp(self) -> str:
        """Get a formatted timestamp for the current time."""
        return datetime.datetime.now().isoformat()

    def set_working_directory(self, directory: str) -> bool:
        """
        Set the current working directory.

        Args:
            directory: The directory to change to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to absolute path if relative
            if not os.path.isabs(directory):
                directory = os.path.normpath(os.path.join(self.current_dir, directory))
            
            # Check if directory exists
            if not os.path.isdir(directory):
                logger.error(f"Directory does not exist: {directory}")
                return False
            
            # Change directory
            os.chdir(directory)
            self.current_dir = directory
            logger.info(f"Changed working directory to {directory}")
            return True
        except Exception as e:
            logger.error(f"Error changing directory: {str(e)}")
            return False

    def get_working_directory(self) -> str:
        """Get the current working directory."""
        return self.current_dir

    def get_env_var(self, name: str) -> Optional[str]:
        """
        Get an environment variable.

        Args:
            name: Name of the environment variable

        Returns:
            Value of the environment variable or None if not found
        """
        return self.env_vars.get(name)

    def set_env_var(self, name: str, value: str) -> None:
        """
        Set an environment variable for the current session.

        Args:
            name: Name of the environment variable
            value: Value to set
        """
        self.env_vars[name] = value
        os.environ[name] = value
        logger.info(f"Set environment variable: {name}={value}")

    def unset_env_var(self, name: str) -> bool:
        """
        Unset an environment variable.

        Args:
            name: Name of the environment variable to unset

        Returns:
            True if successful, False if variable wasn't set
        """
        if name in self.env_vars:
            del self.env_vars[name]
            if name in os.environ:
                del os.environ[name]
            logger.info(f"Unset environment variable: {name}")
            return True
        return False

    def get_all_env_vars(self) -> Dict[str, str]:
        """Get all environment variables."""
        return dict(self.env_vars)

    def set_user_variable(self, name: str, value: Any) -> None:
        """
        Set a user-defined variable in the context.

        Args:
            name: Variable name
            value: Variable value
        """
        self.user_variables[name] = value
        logger.info(f"Set user variable: {name}")

    def get_user_variable(self, name: str) -> Optional[Any]:
        """
        Get a user-defined variable from the context.

        Args:
            name: Variable name

        Returns:
            Variable value or None if not found
        """
        return self.user_variables.get(name)

    def set_preference(self, name: str, value: Any) -> None:
        """
        Set a user preference.

        Args:
            name: Preference name
            value: Preference value
        """
        self.preferences[name] = value
        logger.info(f"Set preference: {name}={value}")

    def get_preference(self, name: str, default: Any = None) -> Any:
        """
        Get a user preference.

        Args:
            name: Preference name
            default: Default value if preference not found

        Returns:
            Preference value or default if not found
        """
        return self.preferences.get(name, default)

    def add_command_to_history(
        self, command: str, result: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a command to the command history.

        Args:
            command: The executed command
            result: The command execution result
            metadata: Additional metadata about the command
        """
        timestamp = self.get_timestamp()
        
        # Create history entry
        history_entry = {
            "command": command,
            "timestamp": timestamp,
            "working_dir": self.current_dir,
            "exit_code": result.get("exit_code"),
        }
        
        if metadata:
            history_entry["metadata"] = metadata
            
        # Add to history and trim if needed
        self.command_history.append(history_entry)
        if len(self.command_history) > self.max_history_size:
            self.command_history = self.command_history[-self.max_history_size:]
            
        # Add command output to output history
        output_entry = {
            "command": command,
            "timestamp": timestamp,
            "output": result.get("output", ""),
            "error": result.get("error", ""),
            "exit_code": result.get("exit_code"),
        }
        
        self.output_history.append(output_entry)
        if len(self.output_history) > self.max_history_size:
            self.output_history = self.output_history[-self.max_history_size:]

    def get_command_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the command execution history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of command history entries
        """
        if limit is None or limit > len(self.command_history):
            return self.command_history
        return self.command_history[-limit:]

    def get_output_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the command output history.

        Args:
            limit: Maximum number of output entries to return

        Returns:
            List of command output entries
        """
        if limit is None or limit > len(self.output_history):
            return self.output_history
        return self.output_history[-limit:]

    def get_last_command_output(self) -> Optional[Dict[str, Any]]:
        """
        Get the output of the last executed command.

        Returns:
            Output entry for the last command or None if no commands executed
        """
        if self.output_history:
            return self.output_history[-1]
        return None

    def clear_history(self) -> None:
        """Clear the command and output history."""
        self.command_history = []
        self.output_history = []
        logger.info("Cleared command and output history")

    def add_plugin_state(self, plugin_name: str, state: Dict[str, Any]) -> None:
        """
        Add or update state for a plugin.

        Args:
            plugin_name: Name of the plugin
            state: Plugin state to store
        """
        self.plugin_states[plugin_name] = state
        logger.info(f"Updated state for plugin: {plugin_name}")

    def get_plugin_state(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get stored state for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Plugin state or None if not found
        """
        return self.plugin_states.get(plugin_name)

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.

        Returns:
            Dictionary with session information
        """
        now = datetime.datetime.now()
        start_time = datetime.datetime.fromisoformat(self.session_start_time)
        duration = (now - start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "duration_seconds": duration,
            "command_count": len(self.command_history),
            "system_info": self.system_info,
            "current_dir": self.current_dir,
        }

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context.

        Returns:
            Dictionary with context summary
        """
        return {
            "session": self.get_session_info(),
            "environment": {
                "working_directory": self.current_dir,
                "path": self.env_vars.get("PATH", ""),
                # Include other relevant environment variables
                "home": self.env_vars.get("HOME", ""),
                "user": self.env_vars.get("USER", ""),
            },
            "recent_commands": self.get_command_history(5),
            "user_variables_count": len(self.user_variables),
            "preferences_count": len(self.preferences),
            "plugins_with_state": list(self.plugin_states.keys()),
        }

    def export_context(self, file_path: Optional[str] = None) -> str:
        """
        Export the current context to a JSON file.

        Args:
            file_path: Path to export to, or None to use a timestamped filename

        Returns:
            Path to the exported file
        """
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(__file__).parent.parent / "data" / f"context_export_{timestamp}.json"
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(

