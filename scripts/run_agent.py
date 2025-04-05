#!/usr/bin/env python3
"""
Agent Mode - LLM-powered terminal assistant with multi-step workflow capabilities.

This script serves as the main entry point for the Agent Mode application.
It configures and launches the interactive agent with workflow support.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.logging import RichHandler

from agent.core import AgentCore
from interface.repl import start_repl, app

# Initialize rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("agent")

# Global variable for the agent core
agent_core = None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Mode - LLM-powered terminal assistant with workflow capabilities"
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (overrides environment variable)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        help="Directory for storing agent data"
    )
    
    # Behavior options
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-color", 
        action="store_true", 
        help="Disable colored output"
    )
    parser.add_argument(
        "--safe-mode", 
        action="store_true", 
        help="Run in safe mode with restricted permissions"
    )
    
    # Command execution
    parser.add_argument(
        "--command", 
        "-c", 
        type=str, 
        help="Execute a single command and exit"
    )
    parser.add_argument(
        "--workflow", 
        "-w", 
        type=str, 
        help="Create and execute a workflow from a request and exit"
    )
    
    return parser.parse_args()


def load_environment_config() -> Dict[str, str]:
    """Load configuration from environment variables."""
    config = {}
    
    # API key
    if "OPENAI_API_KEY" in os.environ:
        config["api_key"] = os.environ["OPENAI_API_KEY"]
    
    # Data directory
    if "AGENT_DATA_DIR" in os.environ:
        config["data_dir"] = os.environ["AGENT_DATA_DIR"]
    
    # Plugins config
    if "AGENT_PLUGINS_CONFIG" in os.environ:
        config["plugins_config"] = os.environ["AGENT_PLUGINS_CONFIG"]
    
    # Log level
    if "AGENT_LOG_LEVEL" in os.environ:
        config["log_level"] = os.environ["AGENT_LOG_LEVEL"]
    
    return config


def load_config_file(config_path: str) -> Dict[str, str]:
    """Load configuration from a file."""
    config = {}
    
    if not config_path or not os.path.exists(config_path):
        return config
    
    try:
        # Simple config file format: key=value
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                if "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
    
    return config


def apply_logging_configuration(verbose: bool):
    """Configure logging based on arguments."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)


def initialize_agent(args, config: Dict[str, str]) -> AgentCore:
    """Initialize the agent core with configuration."""
    # Show initialization message
    with Progress() as progress:
        task = progress.add_task("[cyan]Initializing Agent Mode...", total=100)
        
        # Prepare agent configuration
        agent_config = {}
        
        # API key priority: argument > environment > config file
        if args.api_key:
            agent_config["openai_api_key"] = args.api_key
        elif "api_key" in config:
            agent_config["openai_api_key"] = config["api_key"]
        
        # Data directory
        if args.data_dir:
            agent_config["data_dir"] = args.data_dir
        elif "data_dir" in config:
            agent_config["data_dir"] = config["data_dir"]
        
        # Plugins configuration
        if "plugins_config" in config:
            agent_config["plugins_config"] = config["plugins_config"]
        
        progress.update(task, advance=50)
        
        # Create data directory if specified and doesn't exist
        if "data_dir" in agent_config and not os.path.exists(agent_config["data_dir"]):
            os.makedirs(agent_config["data_dir"])
        
        # Initialize the agent with configuration
        try:
            agent = AgentCore(**agent_config)
            
            # Apply safe mode if requested
            if args.safe_mode:
                agent.permissions["filesystem"]["write"] = False
                agent.permissions["filesystem"]["execute"] = False
                agent.permissions["system"]["execute"] = False
                logger.info("Running in safe mode with restricted permissions")
            
            progress.update(task, advance=50)
            
            return agent
        except Exception as e:
            console.print(f"[bold red]Error initializing agent:[/bold red] {str(e)}")
            sys.exit(1)


def handle_signals():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        if sig == signal.SIGINT:
            console.print("\n[yellow]Interrupted by user. Shutting down...[/yellow]")
        else:
            console.print("\n[red]Received termination signal. Shutting down...[/red]")
        
        # Perform cleanup
        if agent_core:
            # Save any state that needs to be preserved
            pass
            
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination


def display_welcome_message():
    """Display a welcome message with system information."""
    console.print(Panel.fit(
        "[bold green]Agent Mode v0.1.0[/bold green]\n"
        "[cyan]LLM-powered terminal assistant with workflow capabilities[/cyan]\n\n"
        f"[white]System: {sys.platform}[/white]\n"
        f"[white]Python: {sys.version.split()[0]}[/white]\n"
        "[white]Type 'help' for available commands[/white]",
        title="Welcome to Agent Mode",
        border_style="blue"
    ))


def main():
    """Main entry point."""
    global agent_core
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply no-color setting if requested
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    
    # Configure logging
    apply_logging_configuration(args.verbose)
    
    # Load configuration from environment and config file
    config = load_environment_config()
    if args.config:
        config.update(load_config_file(args.config))
    
    # Set up signal handlers
    handle_signals()
    
    # Initialize the agent
    agent_core = initialize_agent(args, config)
    
    # Single command execution mode
    if args.command:
        result = asyncio.run(agent_core.execute_command(args.command))
        if result.output:
            console.print(result.output)
        if result.error:
            console.print(f"[red]{result.error}[/red]")
        return
    
    # Workflow execution mode
    if args.workflow:
        workflow_info = asyncio.run(agent_core.create_workflow(args.workflow))
        if "error" in workflow_info:
            console.print(f"[red]Error creating workflow:[/red] {workflow_info['error']}")
            return
        
        result = asyncio.run(agent_core.execute_workflow(workflow_info["workflow_id"]))
        if "error" in result:
            console.print(f"[red]Error executing workflow:[/red] {result['error']}")
        else:
            console.print(f"[green]Workflow completed with status:[/green] {result['status']}")
        return
    
    # Display welcome message
    display_welcome_message()
    
    # Start the interactive REPL
    try:
        # Import the function from interface/repl.py
        from interface.repl import start_repl
        start_repl(agent_core)
    except Exception as e:
        console.print(f"[bold red]Error starting REPL:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

