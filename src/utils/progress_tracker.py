"""
Progress Tracking Utilities

This module provides utilities for tracking and displaying progress during
research tasks.
"""

import time
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, SpinnerColumn

class ProgressTracker:
    """
    Track and display progress for research tasks.
    
    This class provides utilities for tracking research progress and
    updating status displays.
    """
    
    def __init__(
        self,
        project_dir: str,
        max_steps: int,
        console: Console = None,
        update_interval: int = 5  # Only update UI every 5% progress change
    ):
        """
        Initialize a ProgressTracker instance.
        
        Args:
            project_dir: Directory where project files are stored
            max_steps: Maximum number of steps for the research task
            console: Optional Rich console instance
            update_interval: Minimum progress percentage change before updating UI
        """
        self.project_dir = project_dir
        self.max_steps = max_steps
        self.console = console or Console()
        self.update_interval = update_interval
        
        # Progress state
        self.progress = None
        self.overall_progress = None
        self.current_task = None
        self.status_panel = None
        self.last_progress_percentage = 0
        self.last_update_time = time.time()
        self.status_file_path = os.path.join(project_dir, "status.md")
    
    def update_progress(
        self,
        step: int,
        total: int,
        description: str,
        status: str = "In Progress"
    ) -> None:
        """
        Update the progress display.
        
        Args:
            step: Current step number
            total: Total number of steps
            description: Description of current activity
            status: Current status (e.g., "In Progress", "Completed")
        """
        # Calculate progress percentage
        progress_percentage = min(100, int(step / total * 100))
        
        # Check if update is needed based on progress change or time elapsed
        current_time = time.time()
        progress_change = abs(progress_percentage - self.last_progress_percentage)
        time_elapsed = current_time - self.last_update_time
        
        # Only update if significant progress change or enough time has elapsed
        if progress_change >= self.update_interval or time_elapsed >= 10:  # Update every 10 seconds regardless
            # Update progress bar
            if self.progress and self.overall_progress is not None:
                self.progress.update(self.overall_progress, completed=progress_percentage, description=f"Overall Progress")
            
            # Update current task
            if self.progress and self.current_task is not None:
                self.progress.update(self.current_task, completed=progress_percentage, description=f"Current Task: {description[:60]}" + ("..." if len(description) > 60 else ""))
            
            # Update status panel
            if self.status_panel:
                content = f"[bold]{status}:[/] {description}\n\n"
                content += f"Progress: {progress_percentage}% complete"
                self.status_panel.renderable = Text.from_markup(content)
            
            # Update status file
            self._update_status_file(status, description, progress_percentage)
            
            # Update tracking variables
            self.last_progress_percentage = progress_percentage
            self.last_update_time = current_time
    
    def _update_status_file(self, status: str, description: str, progress_percentage: int) -> None:
        """
        Update the status file with current progress information.
        
        Args:
            status: Current status (e.g., "In Progress", "Completed")
            description: Description of current activity
            progress_percentage: Overall progress percentage
        """
        try:
            with open(self.status_file_path, "w") as f:
                f.write(f"# Research Status\n\n")
                f.write(f"## Current Status: {status}\n\n")
                f.write(f"## Current Task\n{description}\n\n")
                f.write(f"## Progress\n{progress_percentage}%\n\n")
                f.write(f"## Last Updated\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        except Exception as e:
            # Silently handle errors - logging would just create more noise
            pass
