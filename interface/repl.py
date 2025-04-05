# interface/repl.py
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from agent.core import AgentCore
from agent.workflow import WorkflowStatus, StepStatus

# Initialize the console for rich output
console = Console()

# Create the Typer app
app = typer.Typer(help="Agent Mode Terminal Interface with Workflow Support")

# Initialize the agent core
core = AgentCore()

# Main command group
@app.callback()
def callback():
    """Agent Mode Terminal Interface with LLM-powered workflow capabilities."""
    pass


# ---- Direct Command Execution ----

@app.command("chat", help="Process a natural language request")
def process_chat(
    message: str = typer.Argument(..., help="Natural language request to process"),
    direct: bool = typer.Option(False, "--direct", "-d", help="Execute as direct command without workflow")
):
    """Process a natural language request, converting it to a command or workflow."""
    if direct:
        # Execute as direct command
        result = asyncio.run(core.execute_command(message))
        
        if result.exit_code == 0:
            console.print(f"[green]Command executed successfully[/green]")
            if result.output:
                console.print(Panel(result.output, title="Output", border_style="green"))
        else:
            console.print(f"[red]Command failed with exit code {result.exit_code}[/red]")
            if result.error:
                console.print(Panel(result.error, title="Error", border_style="red"))
            if result.output:
                console.print(Panel(result.output, title="Output", border_style="yellow"))
    else:
        # Process as a workflow request
        console.print(f"[bold blue]Processing request:[/bold blue] {message}")
        result = asyncio.run(core.process_input(message))
        
        if result["type"] == "workflow":
            console.print(f"[green]Created workflow:[/green] {result['workflow']['name']}")
            # Display workflow execution result if available
            if "execution_result" in result:
                _display_workflow_result(result["execution_result"])
        else:
            console.print(f"[yellow]Executed as command:[/yellow] {result['command']}")
            cmd_result = result["result"]
            if cmd_result["exit_code"] == 0:
                console.print(f"[green]Command executed successfully[/green]")
                if cmd_result["output"]:
                    console.print(Panel(cmd_result["output"], title="Output", border_style="green"))
            else:
                console.print(f"[red]Command failed with exit code {cmd_result['exit_code']}[/red]")
                if cmd_result["error"]:
                    console.print(Panel(cmd_result["error"], title="Error", border_style="red"))
                if cmd_result["output"]:
                    console.print(Panel(cmd_result["output"], title="Output", border_style="yellow"))


@app.command("list", help="List all active workflows")
def run_command(
    command: str = typer.Argument(..., help="The command to execute")
):
    """Run a shell command directly through the agent."""
    result = asyncio.run(core.execute_command(command))
    
    if result.exit_code == 0:
        console.print(f"[green]Command executed successfully[/green]")
        if result.output:
            console.print(Panel(result.output, title="Output", border_style="green"))
    else:
        console.print(f"[red]Command failed with exit code {result.exit_code}[/red]")
        if result.error:
            console.print(Panel(result.error, title="Error", border_style="red"))
        if result.output:
            console.print(Panel(result.output, title="Output", border_style="yellow"))


# ---- Workflow Management Commands ----

@app.command("workflow", help="Create and execute a workflow from a natural language request")
def create_workflow(
    request: str = typer.Argument(..., help="Natural language request to create a workflow for"),
    execute: bool = typer.Option(True, "--execute/--no-execute", help="Whether to execute the workflow immediately"),
    auto_confirm: bool = typer.Option(False, "--auto-confirm", help="Automatically confirm all steps")
):
    """Create a workflow from a natural language request and optionally execute it."""
    # Create the workflow
    console.print(f"[bold blue]Creating workflow from request:[/bold blue] {request}")
    workflow_info = asyncio.run(core.create_workflow(request))
    
    if "error" in workflow_info:
        console.print(f"[red]Error creating workflow:[/red] {workflow_info['error']}")
        return
    
    # Display workflow info
    _display_workflow_info(workflow_info)
    
    # Execute if requested
    if execute:
        console.print("\n[bold blue]Executing workflow...[/bold blue]")
        result = asyncio.run(core.execute_workflow(workflow_info["workflow_id"], auto_confirm))
        _display_workflow_result(result)


@app.command("execute", help="Execute an existing workflow")
def execute_workflow(
    workflow_id: Optional[str] = typer.Argument(None, help="ID of the workflow to execute (uses active workflow if not specified)"),
    auto_confirm: bool = typer.Option(False, "--auto-confirm", help="Automatically confirm all steps")
):
    """Execute an existing workflow by ID or the currently active workflow."""
    # Check active workflow if ID not provided
    if not workflow_id:
        active_workflow = core.get_active_workflow()
        if not active_workflow:
            console.print("[red]No active workflow. Please specify a workflow ID.[/red]")
            return
        workflow_id = active_workflow["workflow_id"]
    
    console.print(f"[bold blue]Executing workflow {workflow_id}...[/bold blue]")
    result = asyncio.run(core.execute_workflow(workflow_id, auto_confirm))
    _display_workflow_result(result)


@app.command("list", help="List all active workflows")
def list_workflows():
    """List all active workflows in the system."""
    workflows = core.workflow_manager.active_workflows
    
    if not workflows:
        console.print("[yellow]No active workflows.[/yellow]")
        return
    
    # Create a table for displaying workflows
    table = Table(title="Active Workflows")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Steps", style="blue")
    table.add_column("Active", style="yellow")
    
    # Get current active workflow ID
    active_id = core.active_workflow_id
    
    # Add workflows to the table
    for workflow_id, workflow in workflows.items():
        is_active = "✓" if workflow_id == active_id else ""
        steps_info = f"{len(workflow.completed_steps)}/{len(workflow.steps)} completed"
        table.add_row(
            workflow_id[:8] + "...",  # Truncate ID for display
            workflow.name,
            workflow.status.value,
            steps_info,
            is_active
        )
    
    console.print(table)


@app.command("status", help="Show status of the current workflow")
def workflow_status(
    workflow_id: Optional[str] = typer.Argument(None, help="ID of the workflow to show (uses active workflow if not specified)")
):
    """Show the detailed status of a workflow."""
    # Check active workflow if ID not provided
    if not workflow_id:
        active_workflow = core.get_active_workflow()
        if not active_workflow:
            console.print("[red]No active workflow. Please specify a workflow ID.[/red]")
            return
        workflow_id = active_workflow["workflow_id"]
    
    # Get the workflow
    workflow = core.workflow_manager.active_workflows.get(workflow_id)
    if not workflow:
        console.print(f"[red]Workflow {workflow_id} not found.[/red]")
        return
    
    # Create a panel with workflow info
    console.print(Panel(
        f"ID: {workflow.workflow_id}\n"
        f"Name: {workflow.name}\n"
        f"Description: {workflow.description}\n"
        f"Status: {workflow.status.value}\n"
        f"Created: {workflow.created_at}\n"
        f"Steps: {len(workflow.steps)} total, {len(workflow.completed_steps)} completed, {len(workflow.failed_steps)} failed\n"
        f"Current step: {workflow.current_step_id or 'None'}\n",
        title=f"Workflow Status: {workflow.name}",
        border_style="blue"
    ))
    
    # Create a table for steps
    steps_table = Table(title="Workflow Steps")
    steps_table.add_column("ID", style="cyan")
    steps_table.add_column("Name", style="green")
    steps_table.add_column("Status", style="magenta")
    steps_table.add_column("Critical", style="red")
    steps_table.add_column("Confirmation", style="yellow")
    
    # Add steps to the table
    for step in workflow.steps:
        critical = "✓" if step.critical else ""
        confirmation = "✓" if step.requires_confirmation else ""
        
        # Determine row style based on status
        status_style = {
            StepStatus.PENDING: "white",
            StepStatus.RUNNING: "yellow",
            StepStatus.SUCCEEDED: "green",
            StepStatus.FAILED: "red",
            StepStatus.SKIPPED: "blue",
            StepStatus.WAITING_CONFIRMATION: "magenta",
            StepStatus.CANCELLED: "red",
        }.get(step.status, "white")
        
        steps_table.add_row(
            step.step_id.split("_")[-1],  # Just show the step number
            step.name,
            step.status.value,
            critical,
            confirmation,
            style=status_style
        )
    
    console.print(steps_table)
    
    # If there are any waiting steps, prompt for confirmation
    waiting_steps = [step for step in workflow.steps if step.status == StepStatus.WAITING_CONFIRMATION]
    if waiting_steps:
        console.print("\n[bold yellow]Steps waiting for confirmation:[/bold yellow]")
        for step in waiting_steps:
            console.print(f"\nStep {step.step_id.split('_')[-1]}: [bold]{step.name}[/bold]")
            console.print(f"Description: {step.description}")
            
            if Confirm.ask("Confirm this step?"):
                console.print("[green]Confirming step...[/green]")
                result = asyncio.run(core.confirm_workflow_step(step.step_id, True, workflow.workflow_id))
                if "error" in result:
                    console.print(f"[red]Error confirming step:[/red] {result['error']}")
                else:
                    console.print(f"[green]Step confirmed. Result: {result['status']}[/green]")
            else:
                console.print("[red]Rejecting step...[/red]")
                result = asyncio.run(core.confirm_workflow_step(step.step_id, False, workflow.workflow_id))
                if "error" in result:
                    console.print(f"[red]Error rejecting step:[/red] {result['error']}")
                else:
                    console.print(f"[yellow]Step rejected. Result: {result['status']}[/yellow]")


@app.command("confirm", help="Confirm a workflow step that's waiting for confirmation")
def confirm_step(
    step_id: str = typer.Argument(..., help="ID of the step to confirm"),
    workflow_id: Optional[str] = typer.Option(None, "--workflow", "-w", help="ID of the workflow (uses active workflow if not specified)"),
    reject: bool = typer.Option(False, "--reject", "-r", help="Reject the step instead of confirming it")
):
    """Confirm or reject a workflow step that's waiting for user input."""
    confirmed = not reject
    action = "Rejecting" if reject else "Confirming"
    
    console.print(f"[bold]{'[red]' if reject else '[green]'}{action} step {step_id}...[/bold]")
    result = asyncio.run(core.confirm_workflow_step(step_id, confirmed, workflow_id))
    
    if "error" in result:
        console.print(f"[red]Error {action.lower()} step:[/red] {result['error']}")
    else:
        status_color = "yellow" if reject else "green"
        console.print(f"[{status_color}]Step {'rejected' if reject else 'confirmed'}. Result: {result['status']}[/{status_color}]")


@app.command("set", help="Set the active workflow")
def set_active_workflow(
    workflow_id: str = typer.Argument(..., help="ID of the workflow to set as active")
):
    """Set the active workflow for the agent."""
    if workflow_id.lower() == "none" or workflow_id.lower() == "clear":
        # Clear the active workflow
        core.set_active_workflow(None)
        console.print("[yellow]Active workflow cleared.[/yellow]")
    else:
        # Set the specified workflow as active
        success = core.set_active_workflow(workflow_id)
        if success:
            console.print(f"[green]Active workflow set to {workflow_id}.[/green]")
        else:
            console.print(f"[red]Workflow {workflow_id} not found.[/red]")


# ---- Plugin Management ----

@app.command("plugins", help="List available plugins")
def list_plugins():
    """List all loaded plugins in the system."""
    plugins = core.list_plugins()
    
    if not plugins:
        console.print("[yellow]No plugins loaded.[/yellow]")
        return
    
    # Create a table for displaying plugins
    table = Table(title="Available Plugins")
    table.add_column("Name", style="green")
    table.add_column("Version", style="blue")
    table.add_column("Description", style="cyan")
    table.add_column("Enabled", style="magenta")
    
    # Add plugins to the table
    for plugin in plugins:
        enabled = "✓" if plugin["enabled"] else "✗"
        table.add_row(
            plugin["name"],
            plugin["version"],
            plugin["description"],
            enabled
        )
    
    console.print(table)


# ---- Helper Functions ----

def _display_workflow_result(result: Dict[str, Any]):
    """Display the result of a workflow execution."""
    if "error" in result:
        console.print(f"[red]Error executing workflow:[/red] {result['error']}")
        return
    
    status_color = {
        "completed": "green",
        "failed": "red",
        "cancelled": "yellow",
        "waiting_user_input": "magenta",
    }.get(result["status"], "blue")
    
    console.print(Panel(
        f"Status: [{status_color}]{result['status']}[/{status_color}]\n"
        f"Steps completed: {result['steps_completed']}/{result['steps_total']}\n"
        f"Steps failed: {result['steps_failed']}\n"
        f"Started at: {result['started_at']}\n"
        f"Completed at: {result['completed_at'] or 'N/A'}"
    ))
    
    # If there's an execution log, display it
    if "execution_log" in result and result["execution_log"]:
        console.print("\n[bold blue]Execution Log:[/bold blue]")
        for entry in result["execution_log"]:
            timestamp = entry.get("timestamp", "")
            event_type = entry.get("event_type", "")
            console.print(f"[dim]{timestamp}[/dim] [{_get_event_color(event_type)}]{event_type}[/{_get_event_color(event_type)}]")


def _get_event_color(event_type: str) -> str:
    """Get the color for an event type."""
    color_map = {
        "workflow_started": "blue",
        "workflow_completed": "green",
        "workflow_failed": "red",
        "step_started": "yellow",
        "step_completed": "green",
        "step_failed": "red",
        "step_skipped": "cyan",
        "user_confirmation": "magenta",
    }
    return color_map.get(event_type, "white")
    
    # Add steps to the table
    for i, step in enumerate(workflow_info["steps"]):
        critical = "✓" if step["critical"] else ""
        confirmation = "✓" if step["requires_confirmation"] else ""
        
        steps_table.add_row(
            str(i+1),
            step["name"],
            step["description"],
            critical,
            confirmation
        )
    
    console.print(steps_table)


def _display_workflow_result(result: Dict[str, Any]):
    """Display the result of a workflow execution."""
    if "error" in result:
        console.print(f"[red]Error executing workflow:[/red] {result['error']}")
        return
    
    status_color = {
        "completed": "green",
        "failed": "red",
        "cancelled": "yellow",
        "waiting_user_input": "magenta",
    }.get(result["status"], "blue")
    
    console.print(Panel(
        f"Status: [{status_color}]{result['status']}[/{status_color}]\n"
        f"Steps completed: {result['steps_completed']}/{result['steps_total']}\n"
        f"Steps failed: {result['steps_failed']}\n"
        f"Started at: {result['started_at']}\n"
        f"Completed at: {result['completed_at'] or 'N/A'}

