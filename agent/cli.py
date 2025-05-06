import typer
from agent.core import RoleManager
from agent.nlp_parser import parse_command

app = typer.Typer()

def main(role: str = typer.Option(None, help="Set the user role")):
    manager = RoleManager(role)
    typer.echo(f"Current role: {manager.role}")
    if manager.has_permission("write"):
        typer.echo("✅ You have permission to write.")
    else:
        typer.echo("❌ You do NOT have permission to write.")

@app.command()
def parse(text: str, use_llm: bool = False):
    result = parse_command(text, use_llm)
    if result:
        typer.echo(f"Parsed: {result}")
        # Route to plugins if needed
    else:
        typer.echo("❌ Could not understand the command.")

app.command()(main)

if __name__ == "__main__":
    app()
