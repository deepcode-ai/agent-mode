"""
Agent Shell module for interactive command-line interface.

This module provides an AgentShell class that manages the interactive REPL
(Read-Eval-Print Loop) for user interactions with the agent.
"""

import argparse
import cmd
import logging
import os
import re
import shlex
import signal
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "agent_shell.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("agent_shell")


class CommandMode(Enum):
    """Enumeration of shell operation modes."""
    
    DIRECT = "direct"       # Direct command execution
    AGENT = "agent"         # AI-assisted mode
    INTERACTIVE = "interactive"  # Interactive dialogue mode
    PLAN = "plan"           # Plan execution mode


class CommandType(Enum):
    """Types of commands that can be processed by the shell."""
    
    SYSTEM = "system"       # System commands (e.g., exit, help)
    SHELL = "shell"         # Shell commands to be executed
    AGENT = "agent"         # Commands for the agent
    CONFIG = "config"       # Configuration commands
    PLUGIN = "plugin"       # Plugin commands


class AgentShell(cmd.Cmd):
    """
    Interactive shell for agent interactions, providing a REPL interface
    for executing commands, managing state, and interacting with the agent.
    """

    intro = """
    ╔═══════════════════════════════════════════╗
    ║             AGENT MODE SHELL              ║
    ║                                           ║
    ║  Type 'help' for a list of commands.      ║
    ║  Type '!<command>' to execute shell cmds. ║
    ║  Type 'exit' to quit.                     ║
    ╚═══════════════════════════════════════════╝
    """
    
    prompt = "\n[Agent Mode] > "
    
    # Document categories for help system
    doc_header = "Agent Commands (type help <command>):"
    misc_header = "System Commands:"
    undoc_header = "Other Commands:"

    def __init__(
        self,
        cli_executor=None,
        permission_manager=None,
        context_manager=None,
        task_planner=None,
        reasoning_engine=None,
        plugin_manager=None,
        debug_mode: bool = False,
    ):
        """
        Initialize the Agent Shell.

        Args:
            cli_executor: Component for executing shell commands
            permission_manager: Component for checking permissions
            context_manager: Component for tracking context
            task_planner: Component for planning and executing tasks
            reasoning_engine: Component for LLM interactions
            plugin_manager: Component for managing plugins
            debug_mode: Whether to enable debug mode
        """
        super().__init__()
        
        # Store components
        self.cli_executor = cli_executor
        self.permission_manager = permission_manager
        self.context_manager = context_manager
        self.task_planner = task_planner
        self.reasoning_engine = reasoning_engine
        self.plugin_manager = plugin_manager
        
        # Console for rich output
        self.console = Console()
        
        # Shell state
        self.debug_mode = debug_mode
        self.command_mode = CommandMode.DIRECT
        self.last_command = ""
        self.last_output = ""
        self.command_history = []
        self.command_count = 0
        self.running = True
        self.current_plan = None
        
        # Initialize helpers
        self._setup_signal_handlers()
        
        logger.info("AgentShell initialized")
        
        # If debug mode, show component status
        if self.debug_mode:
            self._show_component_status()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful termination."""
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Handle SIGTERM
        signal.signal(signal.SIGTERM, self._handle_terminate)

    def _handle_interrupt(self, sig, frame) -> None:
        """Handle keyboard interrupt (Ctrl+C)."""
        self.console.print("\n[bold red]Operation interrupted by user.[/bold red]")
        
        # If in the middle of a plan execution, ask if user wants to abort
        if self.current_plan:
            self.console.print("[yellow]You have an active plan. What would you like to do?[/yellow]")
            self.console.print("1. Abort plan")
            self.console.print("2. Pause plan")
            self.console.print("3. Continue execution")
            
            choice = input("Your choice [1/2/3]: ").strip()
            if choice == "1":
                self._abort_current_plan()
            elif choice == "2":
                self._pause_current_plan()
            # Option 3 continues execution, no action needed
        else:
            # Just reset the input line
            self.console.print("Use 'exit' to quit the agent shell.")
            
    def _handle_terminate(self, sig, frame) -> None:
        """Handle termination signal."""
        self.console.print("\n[bold red]Received termination signal. Shutting down...[/bold red]")
        self.do_exit("")
        
    def _show_component_status(self) -> None:
        """Display the status of all components in debug mode."""
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        # Check each component
        components = [
            ("CLIExecutor", self.cli_executor),
            ("PermissionManager", self.permission_manager),
            ("ContextManager", self.context_manager),
            ("TaskPlanner", self.task_planner),
            ("ReasoningEngine", self.reasoning_engine),
            ("PluginManager", self.plugin_manager),
        ]
        
        for name, component in components:
            status = "Available" if component else "Not Initialized"
            color = "green" if component else "red"
            table.add_row(name, f"[{color}]{status}[/{color}]")
            
        self.console.print(table)

    def emptyline(self) -> bool:
        """Handle empty line input (do nothing)."""
        return False
        
    def default(self, line: str) -> bool:
        """
        Default command handler for unrecognized commands.
        
        Args:
            line: The input command line
            
        Returns:
            True if successful, False otherwise
        """
        line = line.strip()
        
        # Shell command execution (if line starts with !)
        if line.startswith("!"):
            return self._handle_shell_command(line[1:])
            
        # In agent mode, treat everything as a query to the agent
        if self.command_mode == CommandMode.AGENT:
            return self._handle_agent_query(line)
            
        # In plan mode, check for plan-specific commands
        if self.command_mode == CommandMode.PLAN:
            if line.lower() in ["next", "n"]:
                return self._execute_next_plan_step()
            elif line.lower() in ["abort", "a"]:
                return self._abort_current_plan()
            elif line.lower() in ["pause", "p"]:
                return self._pause_current_plan()
            elif line.lower() in ["resume", "r"]:
                return self._resume_current_plan()
            elif line.lower() in ["status", "s"]:
                return self._show_plan_status()
                
        # Default behavior for unknown commands
        self.console.print(f"[red]Unknown command: {line}[/red]")
        self.console.print("Type 'help' for a list of available commands.")
        return False
        
    def _parse_command(self, line: str) -> Tuple[CommandType, str]:
        """
        Parse the input line to determine command type and content.
        
        Args:
            line: The input command line
            
        Returns:
            Tuple of (command_type, command_content)
        """
        line = line.strip()
        
        # Check for system commands
        system_commands = ["exit", "quit", "help", "mode", "history", "clear"]
        if line in system_commands or line.split()[0] in system_commands:
            return CommandType.SYSTEM, line
            
        # Check for shell command execution
        if line.startswith("!"):
            return CommandType.SHELL, line[1:]
            
        # Check for agent commands
        if line.startswith("@"):
            return CommandType.AGENT, line[1:]
            
        # Check for configuration commands
        if line.startswith("config "):
            return CommandType.CONFIG, line[7:]
            
        # Check for plugin commands
        if line.startswith("plugin "):
            return CommandType.PLUGIN, line[7:]
            
        # Default to agent queries in agent mode
        if self.command_mode == CommandMode.AGENT:
            return CommandType.AGENT, line
            
        # Default to shell commands in direct mode
        return CommandType.SHELL, line
        
    def _handle_shell_command(self, command: str) -> bool:
        """
        Execute a shell command.
        
        Args:
            command: The shell command to execute
            
        Returns:
            True if successful, False otherwise
        """
        if not command.strip():
            self.console.print("[yellow]Empty command[/yellow]")
            return False
            
        # Check if CLI executor is available
        if not self.cli_executor:
            self.console.print("[red]CLI Executor not initialized[/red]")
            return False
            
        try:
            # Log the command
            self.last_command = command
            self.command_history.append(command)
            self.command_count += 1
            
            # Execute the command
            self.console.print(f"[dim]Executing: {command}[/dim]")
            result = self.cli_executor.execute(command)
            
            # Store the result in context
            if self.context_manager:
                self.context_manager.add_command_to_history(
                    command, 
                    result,
                    {"mode": self.command_mode.value}
                )
                
            # Display the output
            if result["output"]:
                # For multiline output, use a panel
                if "\n" in result["output"]:
                    syntax = Syntax(
                        result["output"],
                        "bash",
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=True,
                    )
                    self.console.print(Panel(syntax, title="Output"))
                else:
                    self.console.print(result["output"])
                    
            # Display any errors
            if result["error"]:
                self.console.print(f"[red]{result['error']}[/red]")
                
            # Display exit code if non-zero
            if result["exit_code"] != 0:
                self.console.print(f"[yellow]Command exited with code: {result['exit_code']}[/yellow]")
                
            self.last_output = result["output"]
            return result["exit_code"] == 0
            
        except Exception as e:
            self.console.print(f"[bold red]Error executing command:[/bold red] {str(e)}")
            if self.debug_mode:
                self.console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
            return False
            
    def _handle_agent_query(self, query: str) -> bool:
        """
        Process a query to the agent.
        
        Args:
            query: The user's query
            
        Returns:
            True if successful, False otherwise
        """
        if not query.strip():
            self.console.print("[yellow]Empty query[/yellow]")
            return False
            
        # Check if reasoning engine is available
        if not self.reasoning_engine:
            self.console.print("[red]Reasoning Engine not initialized[/red]")
            return False
            
        try:
            # Log the query
            self.last_command = query
            self.command_history.append(query)
            self.command_count += 1
            
            # Display thinking indicator
            with self.console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                # Prepare context for the query
                context = self._build_context_for_query(query)
                
                # TODO: Process the query with the reasoning engine
                # For now, we'll mock the response
                response = f"I received your query: {query}\n\nThis is a placeholder response since the reasoning engine integration is not yet complete."
                
            # Display the response
            self.console.print(Panel(
                Markdown(response),
                title="Agent Response",
                border_style="green"
            ))
            
            self.last_output = response
            return True
            
        except Exception as e:
            self.console.print(f"[bold red]Error processing query:[/bold red] {str(e)}")
            if self.debug_mode:
                self.console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))
            return False
            
    def _build_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Build context information for a query.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with context information
        """
        context = {
            "query": query,
            "command_mode": self.command_mode.value,
            "last_command": self.last_command,
            "last_output": self.last_output,
        }
        
        # Add context manager information if available
        if self.context_manager:
            # Add working directory
            context["working_directory"] = self.context_manager.get_working_directory()
            
            # Add environment variables
            context["env_vars"] = self.context_manager.get_all_env_vars()
            
            # Add command history
            context["command_history"] = self.context_manager.get_command_history(5)
            
            # Add session info
            context["session_info"] = self.context_manager.get_session_info()
            
        return context
        
    def _execute_next_plan_step(self) -> bool:
        """Execute the next step in the current plan."""
        if not self.current_plan:
            self.console.print("[yellow]No active plan[/yellow]")
            return False
            
        # TODO: Implement plan step execution
        self.console.

