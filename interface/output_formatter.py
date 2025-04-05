"""
Output Formatter module for consistent output styling.

This module provides an OutputFormatter class that handles consistent
formatting of output in various formats with color scheme support.
"""

import json
import logging
import os
import re
import sys
import textwrap
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "output_formatter.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("output_formatter")


class OutputFormat(Enum):
    """Enumeration of supported output formats."""
    
    PLAIN = "plain"      # Plain text without formatting
    RICH = "rich"        # Rich text with ANSI color codes
    JSON = "json"        # JSON formatted output
    MARKDOWN = "markdown"  # Markdown formatted output


class OutputTheme(Enum):
    """Enumeration of available color themes."""
    
    DEFAULT = "default"
    LIGHT = "light"
    DARK = "dark"
    MONOCHROME = "monochrome"
    HIGH_CONTRAST = "high_contrast"


class OutputSection(Enum):
    """Enumeration of output sections for styling."""
    
    HEADER = "header"
    SUBHEADER = "subheader"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CODE = "code"
    COMMAND = "command"
    RESULT = "result"
    URL = "url"
    FILENAME = "filename"
    VARIABLE = "variable"


class OutputFormatter:
    """
    Handles consistent formatting of output with support for multiple
    output formats, color themes, and terminal-aware adjustments.
    """

    def __init__(
        self,
        format_type: Union[OutputFormat, str] = OutputFormat.RICH,
        theme: Union[OutputTheme, str] = OutputTheme.DEFAULT,
        auto_detect_terminal: bool = True,
        max_width: Optional[int] = None,
        context_manager=None,
    ):
        """
        Initialize the Output Formatter.

        Args:
            format_type: Type of output formatting to use
            theme: Color theme to use
            auto_detect_terminal: Whether to auto-detect terminal capabilities
            max_width: Maximum width for output (None for auto-detect)
            context_manager: Component for tracking context
        """
        # Convert string enum values to enum types
        if isinstance(format_type, str):
            format_type = OutputFormat(format_type)
        if isinstance(theme, str):
            theme = OutputTheme(theme)
            
        self.format_type = format_type
        self.theme_name = theme
        self.auto_detect_terminal = auto_detect_terminal
        self.context_manager = context_manager
        
        # Terminal information
        self.terminal_width = max_width or self._detect_terminal_width()
        self.supports_color = self._detect_color_support()
        self.unicode_support = self._detect_unicode_support()
        
        # Create rich console
        self.console = Console(
            width=self.terminal_width,
            theme=self._create_theme(theme),
            highlight=True,
        )
        
        # Output templates
        self.templates = self._load_templates()
        
        logger.info(f"OutputFormatter initialized with format: {format_type.value}, theme: {theme.value}")
        logger.info(f"Terminal width: {self.terminal_width}, color: {self.supports_color}, unicode: {self.unicode_support}")

    def _detect_terminal_width(self) -> int:
        """Detect the width of the terminal."""
        try:
            # Try to get terminal size from os
            columns, _ = os.get_terminal_size(0)
            return columns
        except (OSError, AttributeError):
            try:
                # Alternative method using stty
                import subprocess
                result = subprocess.run(
                    ["stty", "size"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    _, columns = map(int, result.stdout.split())
                    return columns
            except Exception:
                pass
            
            # Fallback to default width
            return 80

    def _detect_color_support(self) -> bool:
        """Detect if the terminal supports color."""
        # Check common environment variables
        if "NO_COLOR" in os.environ:
            return False
            
        if "FORCE_COLOR" in os.environ:
            return True
            
        if "TERM" in os.environ:
            term = os.environ["TERM"]
            if term == "dumb":
                return False
            if "color" in term or term in ["xterm", "rxvt", "screen"]:
                return True
                
        # Check if output is a TTY
        return sys.stdout.isatty()

    def _detect_unicode_support(self) -> bool:
        """Detect if the terminal supports Unicode characters."""
        if "LC_ALL" in os.environ or "LANG" in os.environ:
            env_var = os.environ.get("LC_ALL") or os.environ.get("LANG") or ""
            return "UTF-8" in env_var.upper() or "UTF8" in env_var.upper()
            
        # Default to True for modern terminals
        return True

    def _create_theme(self, theme: OutputTheme) -> Theme:
        """Create a rich theme based on the theme name."""
        # Define color schemes for different themes
        themes = {
            OutputTheme.DEFAULT: {
                "header": "bold bright_blue",
                "subheader": "bold cyan",
                "success": "bold green",
                "error": "bold red",
                "warning": "bold yellow",
                "info": "bright_blue",
                "code": "bright_black",
                "command": "bold green",
                "result": "bright_black",
                "url": "underline blue",
                "filename": "magenta",
                "variable": "yellow",
            },
            OutputTheme.LIGHT: {
                "header": "bold blue",
                "subheader": "bold cyan",
                "success": "bold green",
                "error": "bold red",
                "warning": "bold yellow",
                "info": "blue",
                "code": "black",
                "command": "bold blue",
                "result": "black",
                "url": "underline blue",
                "filename": "magenta",
                "variable": "yellow",
            },
            OutputTheme.DARK: {
                "header": "bold bright_cyan",
                "subheader": "bold bright_blue",
                "success": "bold bright_green",
                "error": "bold bright_red",
                "warning": "bold bright_yellow",
                "info": "bright_white",
                "code": "bright_black",
                "command": "bold bright_green",
                "result": "bright_white",
                "url": "underline bright_blue",
                "filename": "bright_magenta",
                "variable": "bright_yellow",
            },
            OutputTheme.MONOCHROME: {
                "header": "bold",
                "subheader": "bold",
                "success": "bold",
                "error": "bold reverse",
                "warning": "bold",
                "info": "",
                "code": "dim",
                "command": "bold",
                "result": "dim",
                "url": "underline",
                "filename": "italic",
                "variable": "bold",
            },
            OutputTheme.HIGH_CONTRAST: {
                "header": "bold bright_white on blue",
                "subheader": "bold bright_white on cyan",
                "success": "bold bright_white on green",
                "error": "bold bright_white on red",
                "warning": "bold black on yellow",
                "info": "bright_white",
                "code": "bright_white on black",
                "command": "bold bright_green",
                "result": "bright_white",
                "url": "underline bright_blue",
                "filename": "bright_white on magenta",
                "variable": "black on bright_yellow",
            },
        }
        
        # Use the specified theme or fall back to default
        theme_colors = themes.get(theme, themes[OutputTheme.DEFAULT])
        
        # Create rich Theme
        return Theme(theme_colors)

    def _load_templates(self) -> Dict[str, str]:
        """Load output templates."""
        return {
            "header": "{text}",
            "subheader": "{text}",
            "info": "{text}",
            "success": "✓ {text}" if self.unicode_support else "SUCCESS: {text}",
            "error": "✗ {text}" if self.unicode_support else "ERROR: {text}",
            "warning": "⚠ {text}" if self.unicode_support else "WARNING: {text}",
            "command": "$ {text}",
            "code_block": "```{language}\n{code}\n```" if self.format_type == OutputFormat.MARKDOWN else "{code}",
            "panel": "{text}",
        }

    def set_format(self, format_type: Union[OutputFormat, str]) -> None:
        """
        Set the output format type.

        Args:
            format_type: The format to use
        """
        if isinstance(format_type, str):
            format_type = OutputFormat(format_type)
            
        self.format_type = format_type
        logger.info(f"Output format set to: {format_type.value}")

    def set_theme(self, theme: Union[OutputTheme, str]) -> None:
        """
        Set the color theme.

        Args:
            theme: The theme to use
        """
        if isinstance(theme, str):
            theme = OutputTheme(theme)
            
        self.theme_name = theme
        self.console.theme = self._create_theme(theme)
        logger.info(f"Theme set to: {theme.value}")

    def format_text(
        self, text: str, style: Union[OutputSection, str] = None, **kwargs
    ) -> str:
        """
        Format text with the specified style.

        Args:
            text: The text to format
            style: The style to apply
            **kwargs: Additional formatting parameters

        Returns:
            Formatted text
        """
        if not text:
            return ""
            
        # Convert string style to enum
        if isinstance(style, str) and style in [s.value for s in OutputSection]:
            style = OutputSection(style)
            
        # Apply template if available
        if isinstance(style, OutputSection) and style.value in self.templates:
            template_name = style.value
            if template_name in self.templates:
                text = self.templates[template_name].format(text=text, **kwargs)
                
        # Return plain text if plain format
        if self.format_type == OutputFormat.PLAIN:
            return text
            
        # Return colored text using rich
        if style and isinstance(style, OutputSection):
            return f"[{style.value}]{text}[/{style.value}]"
            
        return text

    def format_header(self, text: str) -> str:
        """Format text as a header."""
        return self.format_text(text, OutputSection.HEADER)

    def format_subheader(self, text: str) -> str:
        """Format text as a subheader."""
        return self.format_text(text, OutputSection.SUBHEADER)

    def format_success(self, text: str) -> str:
        """Format text as a success message."""
        return self.format_text(text, OutputSection.SUCCESS)

    def format_error(self, text: str) -> str:
        """Format text as an error message."""
        return self.format_text(text, OutputSection.ERROR)

    def format_warning(self, text: str) -> str:
        """Format text as a warning message."""
        return self.format_text(text, OutputSection.WARNING)

    def format_info(self, text: str) -> str:
        """Format text as an informational message."""
        return self.format_text(text, OutputSection.INFO)

    def format_command(self, command: str) -> str:
        """Format a shell command."""
        return self.format_text(command, OutputSection.COMMAND)

    def format_code(self, code: str, language: str = "") -> str:
        """
        Format a code block.

        Args:
            code: The code to format
            language: The programming language for syntax highlighting

        Returns:
            Formatted code
        """
        if self.format_type == OutputFormat.PLAIN:
            # For plain text, just return indented code
            return textwrap.indent(code, "    ")
            
        elif self.format_type == OutputFormat.MARKDOWN:
            # For Markdown, use fenced code blocks
            return f"```{language}\n{code}\n```"
            
        elif self.format_type == OutputFormat.JSON:
            # For JSON, no special formatting
            return code
            
        # For rich text, we'll let the print method handle syntax highlighting
        return code

    def print(self, text: str, style: Union[OutputSection, str] = None, **kwargs) -> None:
        """
        Print formatted text to the console.

        Args:
            text: The text to print
            style: The style to apply
            **kwargs: Additional formatting parameters
        """
        if self.format_type == OutputFormat.PLAIN:
            # Print plain text
            print(self.format_text(text, style, **kwargs))
            
        elif self.format_type == OutputFormat.JSON:
            # For JSON, format as a simple object
            json_obj = {"type": style.value if style else "text", "content": text}
            if kwargs:
                json_obj["properties"] = kwargs
            print(json.dumps(json_obj))
            
        elif self.format_type == OutputFormat.MARKDOWN:
            # Print as Markdown
            formatted = self.format_text(text, style, **kwargs)
            print(formatted)
            
        else:
            # Use rich for formatted output
            if style == OutputSection.CODE or kwargs.get("language"):
                # Syntax highlighting for code
                syntax = Syntax(
                    text,
                    kwargs.get("language", "text"),
                    theme="monokai",
                    line_numbers=kwargs.get("line_numbers", False),
                    word_wrap=kwargs.get("word_wrap", True),
                )
                self.console.print(syntax)
                
            elif kwargs.get("panel"):
                # Show in panel
                title = kwargs.get("title", "")
                self.console.print(Panel(
                    Text(text, style=style.value if style else ""),
                    title=title,
                    border_style=kwargs.get("border_style", "blue"),
                ))
                
            elif kwargs.get("markdown"):

