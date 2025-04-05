"""
Workflow management module for multi-step LLM-powered automation.

This module provides workflow management capabilities enabling the agent to:
- Break down complex tasks into sequenced steps
- Execute steps with proper dependency tracking
- Maintain context between workflow steps
- Allow user confirmation for critical operations
- Learn from workflow execution patterns
"""

import datetime
import json
import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic

from .context_manager import ContextManager
from .reasoning import ReasoningEngine, PromptTemplate

# Configure logging
logger = logging.getLogger("workflow")

# Type definitions
T = TypeVar('T')


class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING_CONFIRMATION = "waiting_confirmation"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Status of an entire workflow."""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_USER_INPUT = "waiting_user_input"
    PAUSED = "paused"


class WorkflowStep:
    """
    Represents a single step in a multi-step workflow.
    """
    
    def __init__(
        self,
        step_id: str,
        name: str,
        description: str,
        action: Union[str, Callable],
        workflow_id: str,
        params: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        requires_confirmation: bool = False,
        critical: bool = False,
        timeout_seconds: Optional[int] = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ):
        """
        Initialize a workflow step.
        
        Args:
            step_id: Unique identifier for the step
            name: Human-readable name
            description: Detailed description of what the step does
            action: Either a command string or a callable function
            workflow_id: ID of the parent workflow
            params: Parameters for the action
            dependencies: List of step IDs that must complete before this step
            requires_confirmation: Whether user confirmation is required before execution
            critical: Whether this step is critical (failure stops workflow)
            timeout_seconds: Maximum time the step can run
            retry_count: Current retry count
            max_retries: Maximum number of retries on failure
        """
        self.step_id = step_id
        self.name = name
        self.description = description
        self.action = action
        self.workflow_id = workflow_id
        self.params = params or {}
        self.dependencies = dependencies or []
        self.requires_confirmation = requires_confirmation
        self.critical = critical
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.max_retries = max_retries
        
        # Execution state
        self.status = StepStatus.PENDING
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.notes: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "params": self.params,
            "dependencies": self.dependencies,
            "requires_confirmation": self.requires_confirmation,
            "critical": self.critical,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowStep':
        """Create a step from dictionary representation."""
        step = cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data["description"],
            action=data.get("action", ""),  # Note: Can't serialize callables
            workflow_id=data["workflow_id"],
            params=data.get("params", {}),
            dependencies=data.get("dependencies", []),
            requires_confirmation=data.get("requires_confirmation", False),
            critical=data.get("critical", False),
            timeout_seconds=data.get("timeout_seconds"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )
        
        # Set status
        status_value = data.get("status", "pending")
        step.status = StepStatus(status_value)
        
        # Set execution details
        step.started_at = data.get("started_at")
        step.completed_at = data.get("completed_at")
        step.result = data.get("result")
        step.error = data.get("error")
        step.notes = data.get("notes", [])
        
        return step
    
    def add_note(self, note: str) -> None:
        """
        Add a note to the step.
        
        Args:
            note: The note to add
        """
        timestamp = datetime.datetime.now().isoformat()
        self.notes.append(f"[{timestamp}] {note}")
    
    def mark_started(self) -> None:
        """Mark the step as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.datetime.now().isoformat()
    
    def mark_completed(self, result: Dict[str, Any]) -> None:
        """
        Mark the step as completed.
        
        Args:
            result: The result of the step execution
        """
        self.status = StepStatus.SUCCEEDED
        self.completed_at = datetime.datetime.now().isoformat()
        self.result = result
    
    def mark_failed(self, error: str) -> None:
        """
        Mark the step as failed.
        
        Args:
            error: Error message explaining the failure
        """
        self.status = StepStatus.FAILED
        self.completed_at = datetime.datetime.now().isoformat()
        self.error = error
    
    def mark_skipped(self, reason: str) -> None:
        """
        Mark the step as skipped.
        
        Args:
            reason: Reason for skipping the step
        """
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.datetime.now().isoformat()
        self.add_note(f"Skipped: {reason}")
    
    def mark_waiting_confirmation(self) -> None:
        """Mark the step as waiting for user confirmation."""
        self.status = StepStatus.WAITING_CONFIRMATION
    
    def mark_cancelled(self, reason: str) -> None:
        """
        Mark the step as cancelled.
        
        Args:
            reason: Reason for cancellation
        """
        self.status = StepStatus.CANCELLED
        self.completed_at = datetime.datetime.now().isoformat()
        self.add_note(f"Cancelled: {reason}")
    
    def can_execute(self, completed_steps: List[str]) -> bool:
        """
        Check if the step can be executed based on dependencies.
        
        Args:
            completed_steps: List of completed step IDs
            
        Returns:
            True if all dependencies are satisfied
        """
        return all(dep in completed_steps for dep in self.dependencies)
    
    def should_retry(self) -> bool:
        """
        Check if the step should be retried after failure.
        
        Returns:
            True if retry is possible
        """
        return self.status == StepStatus.FAILED and self.retry_count < self.max_retries


class Workflow:
    """
    Represents a multi-step workflow managed by the agent.
    """
    
    def __init__(
        self,
        workflow_id: str,
        name: str,
        description: str,
        user_request: str,
        context_manager: ContextManager,
        steps: Optional[List[WorkflowStep]] = None,
        max_steps: int = 20,
        timeout_seconds: Optional[int] = None,
        created_at: Optional[str] = None,
    ):
        """
        Initialize a workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            name: Human-readable name
            description: Detailed description
            user_request: Original user request that created this workflow
            context_manager: Reference to context manager
            steps: List of workflow steps
            max_steps: Maximum number of steps allowed
            timeout_seconds: Maximum time the workflow can run
            created_at: Creation timestamp
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.user_request = user_request
        self.context_manager = context_manager
        self.steps = steps or []
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.created_at = created_at or datetime.datetime.now().isoformat()
        
        # Execution state
        self.status = WorkflowStatus.PLANNING
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.current_step_id: Optional[str] = None
        self.execution_log: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        
        # Step tracking
        self.completed_steps: List[str] = []
        self.failed_steps: List[str] = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary representation."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "user_request": self.user_request,
            "status": self.status.value,
            "steps": [step.to_dict() for step in self.steps],
            "max_steps": self.max_steps,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_step_id": self.current_step_id,
            "execution_log": self.execution_log,
            "error": self.error,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], context_manager: ContextManager) -> 'Workflow':
        """Create a workflow from dictionary representation."""
        workflow = cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data["description"],
            user_request=data["user_request"],
            context_manager=context_manager,
            max_steps=data.get("max_steps", 20),
            timeout_seconds=data.get("timeout_seconds"),
            created_at=data.get("created_at"),
        )
        
        # Set steps
        workflow.steps = [
            WorkflowStep.from_dict(step_data) for step_data in data.get("steps", [])
        ]
        
        # Set status
        status_value = data.get("status", "planning")
        workflow.status = WorkflowStatus(status_value)
        
        # Set execution details
        workflow.started_at = data.get("started_at")
        workflow.completed_at = data.get("completed_at")
        workflow.current_step_id = data.get("current_step_id")
        workflow.execution_log = data.get("execution_log", [])
        workflow.error = data.get("error")
        workflow.completed_steps = data.get("completed_steps", [])
        workflow.failed_steps = data.get("failed_steps", [])
        
        return workflow
    
    def add_step(self, step: WorkflowStep) -> None:
        """
        Add a step to the workflow.
        
        Args:
            step: The step to add
        """
        # Ensure the step has the correct workflow ID
        step.workflow_id = self.workflow_id
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """
        Get a step by ID.
        
        Args:
            step_id: The step ID to find
            
        Returns:
            The step if found, None otherwise
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_next_executable_steps(self) -> List[WorkflowStep]:
        """
        Get the next steps that are ready to execute.
        
        Returns:
            List of executable steps
        """
        executable_steps = []
        
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.can_execute(self.completed_steps):
                executable_steps.append(step)
        
        return executable_steps
    
    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log an event in the workflow execution log.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            **details,
        }
        
        self.execution_log.append(log_entry)
    
    def mark_started(self) -> None:
        """Mark the workflow as started."""
        self.status = WorkflowStatus.IN_PROGRESS
        self.started_at = datetime.datetime.now().isoformat()
        self.log_event("workflow_started", {"workflow_id": self.workflow_id})
    
    def mark_completed(self) -> None:
        """Mark the workflow as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.datetime.now().isoformat()
        self.log_event("workflow_completed", {"workflow_id": self.workflow_id})
    
    def mark_failed(self, error: str) -> None:
        """
        Mark the workflow as failed.
        
        Args:
            error: Error message explaining the failure
        """
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.datetime.now().isoformat()
        self.error = error
        self.log_event("workflow_failed", {"workflow_id": self

