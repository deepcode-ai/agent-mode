"""
Task planning module for breaking down and executing complex tasks.

This module provides a TaskPlanner class that handles complex task execution
by breaking tasks into steps, validating plans, and tracking execution.
"""

import copy
import json
import logging
import os
import re
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "planner.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("planner")


class PlanStatus(Enum):
    """Possible states for a plan execution."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    PAUSED = "paused"


class StepStatus(Enum):
    """Possible states for a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStep:
    """
    Represents a single step in a task execution plan.
    """

    def __init__(
        self,
        step_id: str,
        description: str,
        command: Optional[str] = None,
        is_critical: bool = False,
        requires_confirmation: bool = False,
        dependencies: Optional[List[str]] = None,
        estimated_duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a plan step.

        Args:
            step_id: Unique identifier for this step
            description: Human-readable description of the step
            command: Command to execute for this step, if applicable
            is_critical: Whether this step is critical for plan success
            requires_confirmation: Whether user confirmation is required
            dependencies: List of step IDs that must complete before this step
            estimated_duration: Estimated execution duration in seconds
            metadata: Additional step metadata
        """
        self.step_id = step_id
        self.description = description
        self.command = command
        self.is_critical = is_critical
        self.requires_confirmation = requires_confirmation
        self.dependencies = dependencies or []
        self.estimated_duration = estimated_duration
        self.metadata = metadata or {}
        
        # Execution tracking
        self.status = StepStatus.PENDING
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.output: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.retry_count = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "command": self.command,
            "is_critical": self.is_critical,
            "requires_confirmation": self.requires_confirmation,
            "dependencies": self.dependencies,
            "estimated_duration": self.estimated_duration,
            "metadata": self.metadata,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output": self.output,
            "error": self.error,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanStep":
        """Create a step from dictionary representation."""
        step = cls(
            step_id=data["step_id"],
            description=data["description"],
            command=data.get("command"),
            is_critical=data.get("is_critical", False),
            requires_confirmation=data.get("requires_confirmation", False),
            dependencies=data.get("dependencies", []),
            estimated_duration=data.get("estimated_duration"),
            metadata=data.get("metadata", {}),
        )
        
        # Restore execution state
        step.status = StepStatus(data.get("status", StepStatus.PENDING.value))
        step.start_time = data.get("start_time")
        step.end_time = data.get("end_time")
        step.output = data.get("output", {})
        step.error = data.get("error")
        step.retry_count = data.get("retry_count", 0)
        
        return step


class TaskPlan:
    """
    Represents a complete task execution plan with multiple steps.
    """

    def __init__(
        self,
        plan_id: str,
        name: str,
        description: str,
        steps: List[PlanStep],
        metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout: Optional[int] = None,
    ):
        """
        Initialize a task plan.

        Args:
            plan_id: Unique identifier for this plan
            name: Short name for the plan
            description: Detailed description of the plan
            steps: List of plan steps
            metadata: Additional plan metadata
            max_retries: Maximum number of retries for failed steps
            timeout: Overall timeout for plan execution in seconds
        """
        self.plan_id = plan_id
        self.name = name
        self.description = description
        self.steps = steps
        self.metadata = metadata or {}
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Execution tracking
        self.status = PlanStatus.NOT_STARTED
        self.current_step_index = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.error: Optional[str] = None
        
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current step being executed."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step to be executed."""
        next_idx = self.current_step_index + 1
        if next_idx < len(self.steps):
            return self.steps[next_idx]
        return None
    
    def get_progress(self) -> Tuple[int, int]:
        """Get the current progress as (completed_steps, total_steps)."""
        completed = sum(1 for step in self.steps if step.status == StepStatus.COMPLETED)
        return completed, len(self.steps)
    
    def get_progress_percentage(self) -> float:
        """Get the current progress as a percentage."""
        completed, total = self.get_progress()
        return (completed / total) * 100 if total > 0 else 0
    
    def is_complete(self) -> bool:
        """Check if the plan is complete."""
        return self.status == PlanStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": self.metadata,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPlan":
        """Create a plan from dictionary representation."""
        steps = [PlanStep.from_dict(step_data) for step_data in data["steps"]]
        
        plan = cls(
            plan_id=data["plan_id"],
            name=data["name"],
            description=data["description"],
            steps=steps,
            metadata=data.get("metadata", {}),
            max_retries=data.get("max_retries", 3),
            timeout=data.get("timeout"),
        )
        
        # Restore execution state
        plan.status = PlanStatus(data.get("status", PlanStatus.NOT_STARTED.value))
        plan.current_step_index = data.get("current_step_index", 0)
        plan.start_time = data.get("start_time")
        plan.end_time = data.get("end_time")
        plan.error = data.get("error")
        
        return plan


class PlanValidationError(Exception):
    """Exception raised when a plan fails validation."""
    pass


class PlanExecutionError(Exception):
    """Exception raised when a plan fails during execution."""
    pass


class TaskPlanner:
    """
    Handles planning and execution of complex tasks by breaking them down
    into smaller steps and executing them in sequence with proper error handling.
    """

    def __init__(
        self,
        cli_executor=None,
        permission_manager=None,
        context_manager=None,
        plans_dir: Optional[str] = None,
    ):
        """
        Initialize the Task Planner.

        Args:
            cli_executor: Component for executing shell commands
            permission_manager: Component for checking permissions
            context_manager: Component for tracking context
            plans_dir: Directory to store plan files
        """
        self.cli_executor = cli_executor
        self.permission_manager = permission_manager
        self.context_manager = context_manager
        self.plans_dir = plans_dir or Path(__file__).parent.parent / "data" / "plans"
        
        # Create plans directory if it doesn't exist
        os.makedirs(self.plans_dir, exist_ok=True)
        
        # Currently active plan
        self.active_plan: Optional[TaskPlan] = None
        
        # Registered plan templates
        self.plan_templates: Dict[str, Dict[str, Any]] = {}
        
        # Handlers for specific step types
        self.step_handlers: Dict[str, Callable] = {}
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        
        logger.info("TaskPlanner initialized")
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()

    def _register_default_recovery_strategies(self) -> None:
        """Register default recovery strategies for common errors."""
        # Retry strategy
        def retry_strategy(step: PlanStep, error: Exception) -> Dict[str, Any]:
            if step.retry_count < self.active_plan.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                logger.info(f"Retrying step {step.step_id} (attempt {step.retry_count})")
                return {"action": "retry", "message": f"Retrying step (attempt {step.retry_count})"}
            return {"action": "fail", "message": "Max retry attempts reached"}
            
        # Skip strategy
        def skip_strategy(step: PlanStep, error: Exception) -> Dict[str, Any]:
            if not step.is_critical:
                step.status = StepStatus.SKIPPED
                logger.info(f"Skipping non-critical step {step.step_id}")
                return {"action": "skip", "message": "Step skipped"}
            return {"action": "fail", "message": "Cannot skip critical step"}
            
        # Interactive fix strategy
        def interactive_fix_strategy(step: PlanStep, error: Exception) -> Dict[str, Any]:
            # In a real implementation, this would prompt the user
            logger.info(f"Interactive fix required for step {step.step_id}")
            return {"action": "prompt", "message": f"Error in step: {error}. Please fix manually."}
        
        self.recovery_strategies["retry"] = retry_strategy
        self.recovery_strategies["skip"] = skip_strategy
        self.recovery_strategies["interactive"] = interactive_fix_strategy

    def create_plan(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """
        Create a new task execution plan.

        Args:
            name: Short name for the plan
            description: Detailed description of the plan
            steps: List of step definitions
            metadata: Additional plan metadata

        Returns:
            The created TaskPlan
        """
        plan_id = f"plan_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Convert step dictionaries to PlanStep objects
        plan_steps = []
        for i, step_data in enumerate(steps):
            step_id = step_data.get("step_id", f"step_{i+1}")
            plan_steps.append(
                PlanStep(
                    step_id=step_id,
                    description=step_data["description"],
                    command=step_data.get("command"),
                    is_critical=step_data.get("is_critical", False),
                    requires_confirmation=step_data.get("requires_confirmation", False),
                    dependencies=step_data.get("dependencies", []),
                    estimated_duration=step_data.get("estimated_duration"),
                    metadata=step_data.get("metadata", {}),
                )
            )
        
        # Create the plan
        plan = TaskPlan(
            plan_id=plan_id,
            name=name,
            description=description,
            steps=plan_steps,
            metadata=metadata or {},
            max_retries=metadata.get("max_retries", 3) if metadata else 3,
            timeout=metadata.get("timeout") if metadata else None,
        )
        
        # Validate the plan
        self._validate_plan(plan)
        
        logger.info(f"Created plan: {name} with {len(plan_steps)} steps")
        return plan

    def _validate_plan(self, plan: TaskPlan) -> None:
        """
        Validate a plan for correctness and security.

        Args:
            plan: The plan to validate

        Raises:
            PlanValidationError: If the plan fails validation
        """
        # Check for empty plan
        if not plan.steps:
            raise PlanValidationError("Plan contains no steps")
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            raise PlanValidationError("Plan contains duplicate step IDs")
        
        # Check for dependency cycles and missing dependencies
        dependency_map = {step.step_id: set(step.dependencies) for step in plan.steps}
        all_step_ids = set(step_ids)
        
        # Check for missing dependencies
        

