import os
import time

from rich.console import Console
from rich.table import Table
from rich.progress import Progress


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_training_progress(trainer, start_time):

    console1 = Console()
    console1.rule("[bold red]Training Progress[/bold red]")
    
    console1.print(f"[bold yellow]Train Loss:[/bold yellow] {trainer.loss:.4f}")
    console1.print(f"[bold yellow]Step:[/bold yellow] {trainer.total_steps}")
    iter_time = (time.time() - start_time) / trainer.total_steps
    console1.print(f"[bold yellow]Iter Time:[/bold yellow] {iter_time:.4f} seconds")
    
    console1.rule()


def log_validation_metrics(epoch, acc, ap):

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Value")

    table.add_row("Epoch", f"{epoch}")
    table.add_row("Accuracy", f"{acc:.4f}")
    table.add_row("AP", f"{ap:.4f}")

    console.print(table)