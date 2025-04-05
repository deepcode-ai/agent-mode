    async def execute_command(self, command: str) -> CommandResult:
        """
        Execute a shell command with proper permission checks.
        
        Args:
            command: The command to execute
            
        Returns:
            Command execution result
        """
        # Record the command in history
        cmd_entry = {
            "command": command,
            "timestamp": self.context_manager.get_current_time(),
            "id": str(uuid.uuid4()),
        }
        self.command_history.append(cmd_entry)
        
        # Trim history if needed
        if len(self.command_history) > self.max_history:
            self.command_history = self.command_history[-self.max_history:]
        
        # Check permissions (simplified for now)
        if not self._check_command_permissions(command):
            return CommandResult(
                command=command,
                exit_code=1,
                output="",
                error="Permission denied: This command requires elevated permissions",
                duration=0,
            )
        
        # Execute the command
        result = await self.cli_executor.execute(command)
        
        # Update command history with result
        cmd_entry["exit_code"] = result.exit_code
        cmd_entry["duration"] = result.duration
        
        # Update context with command result
        self.context_manager.update_command_context(command, result)
        
        return result
    
    def _check_command_permissions(self, command: str) -> bool:
        """
        Check if the command is allowed by the current permissions.
        
        Args:
            command: The command to check
            
        Returns:
            True if the command is allowed
        """
        # This is a simplified permission check
        # In a production system, this would be more sophisticated
        
        # Check for dangerous commands
        dangerous_patterns = [
            "rm -rf /", 
            "mkfs", 
            "dd if=", 
            "chmod -R 777", 
            "> /dev/",
            ":(){ :|:& };:",  # Fork bomb
        ]
        
        for pattern in dangerous_patterns:
            if pattern in command:
                return False
        
        # Check for system commands that require permissions
        if self.permissions["system"]["execute"] is False:
            system_commands = ["shutdown", "reboot", "systemctl", "service"]
            if any(cmd in command for cmd in system_commands):
                return False
        
        return True
    
    async def get_llm_response(self, system_message: str, user_message: str) -> str:
        """
        Get a response from the LLM.
        
        Args:
            system_message: The system message
            user_message: The user message
            
        Returns:
            LLM response text
        """
        # This is a placeholder for LLM integration
        # In a real implementation, this would use the OpenAI API
        
        # For now, mock a response based on the messages
        # This will be replaced with actual API calls
        response = f"Response to: {user_message[:30]}..."
        
        return response
    
    # ---- Workflow Management Methods ----
    
    async def create_workflow(self, user_request: str) -> Dict[str, Any]:
        """
        Create a new workflow from a user request.
        
        Args:
            user_request: The user's natural language request
            
        Returns:
            Workflow information
        """
        try:
            workflow = await self.workflow_manager.create_workflow_from_request(user_request)
            self.active_workflow_id = workflow.workflow_id
            
            return {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": [
                    {
                        "id": step.step_id,
                        "name": step.name,
                        "description": step.description,
                        "requires_confirmation": step.requires_confirmation,
                        "critical": step.critical,
                    }
                    for step in workflow.steps
                ],
                "status": workflow.status.value,
            }
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            return {
                "error": f"Failed to create workflow: {str(e)}",
                "status": "failed",
            }
    
    async def execute_workflow(self, workflow_id: Optional[str] = None, auto_confirm: bool = False) -> Dict[str, Any]:
        """
        Execute a workflow.
        
        Args:
            workflow_id: The ID of the workflow to execute (uses active workflow if None)
            auto_confirm: Whether to automatically confirm all steps
            
        Returns:
            Execution result summary
        """
        try:
            # Use active workflow if not specified
            workflow_id = workflow_id or self.active_workflow_id
            if not workflow_id:
                return {
                    "error": "No workflow specified or active",
                    "status": "failed",
                }
            
            # Execute the workflow
            result = await self.workflow_manager.execute_workflow(workflow_id, auto_confirm)
            return result
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}")
            return {
                "error": f"Failed to execute workflow: {str(e)}",
                "status": "failed",
            }
    
    async def confirm_workflow_step(self, step_id: str, confirmed: bool, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Confirm or reject a workflow step.
        
        Args:
            step_id: The ID of the step to confirm
            confirmed: Whether the step is confirmed (True) or rejected (False)
            workflow_id: The ID of the workflow (uses active workflow if None)
            
        Returns:
            Result of the confirmation
        """
        try:
            # Use active workflow if not specified
            workflow_id = workflow_id or self.active_workflow_id
            if not workflow_id:
                return {
                    "error": "No workflow specified or active",
                    "status": "failed",
                }
            
            # Confirm or reject the step
            result = await self.workflow_manager.confirm_step(workflow_id, step_id, confirmed)
            return result
        except Exception as e:
            logger.error(f"Error confirming workflow step: {str(e)}")
            return {
                "error": f"Failed to confirm workflow step: {str(e)}",
                "status": "failed",
            }
    
    def get_active_workflow(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the active workflow.
        
        Returns:
            Active workflow information or None if no active workflow
        """
        if not self.active_workflow_id:
            return None
            
        try:
            workflow = self.workflow_manager.active_workflows.get(self.active_workflow_id)
            if not workflow:
                return None
                
            return {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "status": workflow.status.value,
                "current_step": workflow.current_step_id,
                "steps_total": len(workflow.steps),
                "steps_completed": len(workflow.completed_steps),
                "steps_failed": len(workflow.failed_steps),
            }
        except Exception as e:
            logger.error(f"Error getting active workflow: {str(e)}")
            return None
    
    def set_active_workflow(self, workflow_id: Optional[str]) -> bool:
        """
        Set the active workflow.
        
        Args:
            workflow_id: The ID of the workflow to set as active (None to clear)
            
        Returns:
            True if successful
        """
        if workflow_id is None:
            self.active_workflow_id = None
            return True
            
        if workflow_id in self.workflow_manager.active_workflows:
            self.active_workflow_id = workflow_id
            return True
            
        return False
    
    # ---- Command Processing Methods ----
    
    async def process_input(self, user_input: str, use_workflows: bool = True) -> Dict[str, Any]:
        """
        Process user input, determining whether to execute directly or create a workflow.
        
        Args:
            user_input: The user's input text
            use_workflows: Whether to use workflows for complex requests
            
        Returns:
            Processing result
        """
        # If input starts with '!', it's a direct command
        if user_input.startswith('!'):
            command = user_input[1:].strip()
            result = await self.execute_command(command)
            return {
                "type": "command",
                "command": command,
                "result": {
                    "exit_code": result.exit_code,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration,
                }
            }
        
        # If not using workflows, treat as direct command
        if not use_workflows:
            result = await self.execute_command(user_input)
            return {
                "type": "command",
                "command": user_input,
                "result": {
                    "exit_code": result.exit_code,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration,
                }
            }
        
        # Otherwise, create and execute a workflow
        workflow_info = await self.create_workflow(user_input)
        
        if "error" in workflow_info:
            # Workflow creation failed, fall back to direct execution
            result = await self.execute_command(user_input)
            return {
                "type": "command_fallback",
                "command": user_input,
                "result": {
                    "exit_code": result.exit_code,
                    "output": result.output,
                    "error": result.error,
                    "duration": result.duration,
                }
            }
        
        # Start executing the workflow
        execution_result = await self.execute_workflow(workflow_info["workflow_id"])
        
        return {
            "type": "workflow",
            "workflow": workflow_info,
            "execution_result": execution_result,
        }
    
    # ---- Plugin System Methods ----
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all loaded plugins.
        
        Returns:
            List of plugin information
        """
        return [
            {
                "name": plugin.name,
                "description": getattr(plugin, "description", "No description"),
                "version": getattr(plugin, "version", "0.1.0"),
                "enabled": getattr(plugin, "enabled", True),
            }
            for plugin in self.plugins
        ]
    
    def get_plugin_by_name(self, name: str) -> Optional[Any]:
        """
        Get a plugin by name.
        
        Args:
            name: The name of the plugin
            
        Returns:
            Plugin object or None if not found
        """
        for plugin in self.plugins:
            if plugin.name == name:
                return plugin
        return None
    
    # ---- Utility Methods ----
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Status information
        """
        return {
            "plugins_loaded": len(self.plugins),
            "workflows_active": len(self.workflow_manager.active_workflows),
            "current_workflow": self.get_active_workflow(),
            "command_history_size": len(self.command_history),
            "permissions": self.permissions,
        }

# agent/core.py
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .plugin_loader import PluginLoader
from .context_manager import ContextManager
from .workflow_manager import WorkflowManager
from .cli_executor import CommandExecutor, CommandResult
from .reasoning import ReasoningEngine

# Configure logging
logger = logging.getLogger("agent_core")

class AgentCore:
    """
    Core agent class that integrates all components of the system.
    Responsible for handling commands, workflows, and plugin integration.
    """
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        context_dir: Optional[str] = None,
        workflows_dir: Optional[str] = None,
        plugins_config: Optional[str] = None,
        max_history: int = 100,
    ):
        """
        Initialize the agent core.
        
        Args:
            openai_api_key: API key for OpenAI (uses env var if not provided)
            context_dir: Directory for context storage
            workflows_dir: Directory for workflow storage
            plugins_config: Path to plugins configuration
            max_history: Maximum number of commands to keep in history
        """
        # Initialize API key from args or environment
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Create data directories
        self.data_dir = Path(__file__).parent.parent / "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Components initialization
        self.context_manager = ContextManager(
            context_dir=context_dir or str(self.data_dir / "context")
        )
        self.cli_executor = CommandExecutor(context_manager=self.context_manager)
        self.reasoning_engine = ReasoningEngine(
            api_wrapper=None,  # We'll implement this later
            context_manager=self.context_manager
        )
        
        # Initialize plugin system
        self.plugin_loader = PluginLoader(plugins_config or "config/plugins.yaml")
        self.plugins = self.plugin_loader.load_plugins()
        
        # Initialize workflow manager
        self.workflow_manager = WorkflowManager(
            agent_core=self,
            context_manager=self.context_manager,
            reasoning_engine=self.reasoning_engine,
            workflows_dir=workflows_dir or str(self.data_dir / "workflows"),
        )
        
        # Command history
        self.command_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        
        # Permissions system
        self.permissions = {
            "filesystem": {
                "read": True,
                "write": False,
                "execute": False,
            },
            "network": {
                "outbound": False,
                "inbound": False,
            },
            "system": {
                "read": True,
                "write": False,
                "execute": False,
            }
        }
        
        # Active workflow tracking
        self.active_workflow_id: Optional[str] = None
        
        logger.info("AgentCore initialized")

