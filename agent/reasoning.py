"""
Reasoning module for LLM interactions and self-correction.

This module provides a ReasoningEngine class that manages interactions with LLMs,
handles prompt engineering, and implements self-correction mechanisms.
"""

import json
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "reasoning.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("reasoning")


class ResponseValidationError(Exception):
    """Exception raised when an LLM response fails validation."""
    pass


class PromptTemplate:
    """
    Represents a template for prompting an LLM with dynamic variables.
    """

    def __init__(
        self,
        template_id: str,
        system_message: str,
        user_message: str,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a prompt template.

        Args:
            template_id: Unique identifier for this template
            system_message: System message template
            user_message: User message template
            variables: List of variable names used in the templates
            metadata: Additional template metadata
        """
        self.template_id = template_id
        self.system_message = system_message
        self.user_message = user_message
        self.variables = variables or []
        self.metadata = metadata or {}
        
        # Validate that all variables are used in the templates
        self._validate_variables()
        
    def _validate_variables(self) -> None:
        """Validate that all declared variables are used in the templates."""
        used_vars = set()
        
        # Extract variables using regex to find ${variable} patterns
        for template_str in [self.system_message, self.user_message]:
            found_vars = re.findall(r'\$\{([a-zA-Z0-9_]+)\}', template_str)
            used_vars.update(found_vars)
            
        # Check if any declared variables are not used
        unused_vars = set(self.variables) - used_vars
        if unused_vars:
            logger.warning(f"Template {self.template_id} has unused variables: {unused_vars}")
            
        # Check if any used variables are not declared
        undeclared_vars = used_vars - set(self.variables)
        if undeclared_vars:
            logger.warning(f"Template {self.template_id} has undeclared variables: {undeclared_vars}")
            self.variables.extend(undeclared_vars)
        
    def format(self, variables: Dict[str, str]) -> Tuple[str, str]:
        """
        Format the template with the provided variables.

        Args:
            variables: Dictionary of variable values

        Returns:
            Tuple of formatted (system_message, user_message)
        """
        # Create Template objects
        system_template = Template(self.system_message)
        user_template = Template(self.user_message)
        
        # Format templates
        try:
            formatted_system = system_template.safe_substitute(variables)
            formatted_user = user_template.safe_substitute(variables)
            return formatted_system, formatted_user
        except KeyError as e:
            logger.error(f"Missing variable in template {self.template_id}: {e}")
            raise ValueError(f"Missing required variable: {e}")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            "template_id": self.template_id,
            "system_message": self.system_message,
            "user_message": self.user_message,
            "variables": self.variables,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create a template from dictionary representation."""
        return cls(
            template_id=data["template_id"],
            system_message=data["system_message"],
            user_message=data["user_message"],
            variables=data.get("variables", []),
            metadata=data.get("metadata", {}),
        )


class ResponseValidator:
    """
    Handles validation of LLM responses to ensure they're safe and appropriate.
    """

    def __init__(self):
        """Initialize the response validator."""
        # Register validators
        self.validators: Dict[str, Callable[[str], Tuple[bool, Optional[str]]]] = {
            "json_schema": self._validate_json_schema,
            "code_safety": self._validate_code_safety,
            "toxic_content": self._validate_toxic_content,
            "command_safety": self._validate_command_safety,
        }

    def validate(
        self, response: str, validation_types: List[str], schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an LLM response.

        Args:
            response: The response text to validate
            validation_types: List of validation types to apply
            schema: JSON schema for validation (if applicable)

        Returns:
            Tuple of (is_valid, error_message)
        """
        for validation_type in validation_types:
            if validation_type in self.validators:
                is_valid, error_message = self.validators[validation_type](response, schema)
                if not is_valid:
                    return False, error_message
            else:
                logger.warning(f"Unknown validation type: {validation_type}")
        
        return True, None
        
    def _validate_json_schema(
        self, response: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that the response adheres to a JSON schema."""
        if not schema:
            return True, None
            
        try:
            # Try to parse the response as JSON
            response_json = json.loads(response)
            
            # Basic schema validation
            # In a production implementation, use a proper JSON Schema validator
            for key, value_type in schema.items():
                if key not in response_json:
                    return False, f"Missing required key: {key}"
                
                if value_type == "string" and not isinstance(response_json[key], str):
                    return False, f"Key {key} should be a string"
                elif value_type == "number" and not isinstance(response_json[key], (int, float)):
                    return False, f"Key {key} should be a number"
                elif value_type == "boolean" and not isinstance(response_json[key], bool):
                    return False, f"Key {key} should be a boolean"
                elif value_type == "array" and not isinstance(response_json[key], list):
                    return False, f"Key {key} should be an array"
                elif value_type == "object" and not isinstance(response_json[key], dict):
                    return False, f"Key {key} should be an object"
            
            return True, None
        except json.JSONDecodeError:
            return False, "Invalid JSON format"
            
    def _validate_code_safety(
        self, response: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that code responses don't contain unsafe patterns."""
        # Check for potentially harmful code patterns
        dangerous_patterns = [
            r"rm\s+-rf\s+/",  # Recursive deletion of root
            r"sudo\s+rm",     # Sudo removal
            r"system\([^)]*rm", # System calls with rm
            r"eval\(",        # Eval functions
            r"exec\(",        # Exec functions
            r"os\.system\(",  # OS system calls
            r"subprocess\.call\(", # Subprocess calls
            r"__import__\(",  # Dynamic imports
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False, f"Response contains potentially unsafe code pattern: {pattern}"
        
        return True, None
        
    def _validate_toxic_content(
        self, response: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that the response doesn't contain toxic or inappropriate content."""
        # Basic check for inappropriate content
        # In a production implementation, use a dedicated toxic content detector
        toxic_patterns = [
            r"fuck",
            r"shit",
            r"asshole",
            r"bitch",
            r"cunt",
            r"damn",
            r"nigger",
            r"faggot",
        ]
        
        for pattern in toxic_patterns:
            if re.search(r"\b" + pattern + r"\b", response, re.IGNORECASE):
                return False, "Response contains inappropriate language"
        
        return True, None
        
    def _validate_command_safety(
        self, response: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that shell commands in the response are safe."""
        # Check for potentially dangerous shell commands
        dangerous_commands = [
            r"rm\s+-rf",
            r"mkfs",
            r"dd\s+if=.*\s+of=/dev/",
            r"sudo",
            r"su\s+-",
            r"chmod\s+-R\s+777",
            r"chown\s+-R",
            r":(){:\|:&};:",  # Fork bomb
            r">(\/dev\/sd[a-z])",  # Disk writes
            r"shutdown",
            r"reboot",
            r"halt",
        ]
        
        shell_command_blocks = re.findall(r"```(?:bash|sh)(.*?)```", response, re.DOTALL)
        if not shell_command_blocks:
            shell_command_blocks = re.findall(r"`([^`]*)`", response)
        
        for command_block in shell_command_blocks:
            for pattern in dangerous_commands:
                if re.search(pattern, command_block, re.IGNORECASE):
                    return False, f"Response contains potentially dangerous command: {pattern}"
        
        return True, None


class ReasoningEngine:
    """
    Manages interactions with LLMs, handles prompt engineering,
    and implements self-correction mechanisms.
    """

    def __init__(
        self,
        api_wrapper=None,
        templates_dir: Optional[str] = None,
        max_retries: int = 3,
        context_manager=None,
    ):
        """
        Initialize the Reasoning Engine.

        Args:
            api_wrapper: Component for API interactions
            templates_dir: Directory containing prompt templates
            max_retries: Maximum number of retry attempts for failed requests
            context_manager: Component for tracking context
        """
        # Store components
        self.api_wrapper = api_wrapper
        self.context_manager = context_manager
        self.templates_dir = templates_dir or Path(__file__).parent.parent / "config" / "templates"
        self.max_retries = max_retries
        
        # Create templates directory if it doesn't exist
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Initialize prompt templates
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Initialize response validator
        self.validator = ResponseValidator()
        
        # Load default templates
        self._load_default_templates()
        
        # Load any custom templates from disk
        self._load_templates_from_disk()
        
        logger.info("ReasoningEngine initialized")

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        default_templates = [
            PromptTemplate(
                template_id="command_generation",
                system_message="""You are an AI assistant that helps users by generating shell commands.
You should generate commands that are safe, efficient, and appropriate for the user's operating system.
Always explain what the command does before providing it. Format commands between backticks like:
`command here`""",
                user_message="I need a command to ${task}. My operating system is ${os}.",
                variables=["task", "os"],
                metadata={"purpose": "Generate shell commands"},
            ),
            PromptTemplate(
                template_id="plan_creation",
                system_message="""You are an AI assistant that helps users by creating detailed plans.
Break down the task into logical steps, considering dependencies between steps.
For each step, include a clear description and any relevant commands.""",
                user_message="I need a plan to ${task}. The context is: ${context}",
                variables=["task", "context"],
                metadata={"purpose": "Create execution plans"},
            ),
            PromptTemplate(
                template_id="error_analysis",
                system_message="""You are an AI assistant that analyzes errors and suggests solutions.
Provide clear explanations of what went wrong and actionable steps to fix the issue.""",
                user_message="I encountered this error while trying to ${task}:\n```\n${error}\n```\nThe command I ran was: `${command}`",
                variables=["task", "error", "command"],
                metadata={"purpose": "Analyze errors and suggest fixes"},
            ),
            PromptTemplate(
                template_id="self_correction",
                system_message="""You are an AI assistant that reflects on and corrects your previous response.
Identify any errors or omissions in your previous response and provide a corrected version.""",
                user_message="""Your previous response was:
```
${previous_response}
```

This response had the following issue: ${issue}

Please provide a corrected response.""",
                variables=["previous_response", "issue"],
                metadata={"purpose": "Self-correction"},
            ),
        ]
        
        for template in default_templates:
            self.templates[template.template_id] = template
            
        logger.info(f"Loaded {len(default_templates)} default templates")

    def _load_templates_from_disk(self) -> None:
        """Load prompt templates from the templates directory."""
        try:
            if os.path.exists(self.templates_dir):
                template_files = [f for f in os.listdir(self.templates_dir) if f.endswith('.json')]
                
                for file_name in template_files:
                    try:
                        file_path = os.path.join(self.templates_dir, file_name)
                        with open(file_path, 'r') as f:
                            template_data = json.load(f)
                            template = PromptTemplate.from_dict(template_data)
                            self.templates[template.template_id] = template
                            logger.info(f"Loaded template from {file_path}")
                    except Exception as e:
                        logger.error(f"Error loading template from {file_name}: {str(e)}")
        except Exception as e:
            logger.error(

