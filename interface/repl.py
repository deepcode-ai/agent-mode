# interface/repl.py
import typer
from agent.core import AgentCore

app = typer.Typer()
core = None

@app.callback()
def main(role: str = typer.Option("default", help="User role")):
    global core
    core = AgentCore(user_role=role)

@app.command()
def run(command: str):
    core.execute_command(command)

@app.command()
def list_plugins():
    for plugin in core.plugins:
        print(f"- {plugin.name}")
