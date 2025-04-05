"""
Workflow Manager for LLM-powered multi-step automation.

This module orchestrates workflow creation, execution, and monitoring, enabling:
- Automatic breakdown of complex user requests into executable steps
- LLM-powered planning and decision-making
- User confirmation for critical operations
- Error recovery and workflow adjustment
- Persistent workflow state management
"""

import datetime
import json
import logging
import os
import re
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar

from .context_manager import ContextManager
from .reasoning import ReasoningEngine, PromptTemplate, ResponseValidationError
from .workflow import Workflow, WorkflowStep, WorkflowStatus, StepStatus
from .core import AgentCore
from .cli_executor import CommandResult

# Configure logging
logger = logging.getLogger("workflow_manager")

# Define workflow template prompts
WORKFLOW_PLANNING_PROMPT = """
You are an expert AI workflow planner. Your task is to break down a complex user request into a series of 
executable steps that can be performed in sequence.

User request: ${user_request}

System context:
${system_context}

Your task is to:
1. Analyze the request to understand the user's goal
2. Break it down into 2-10 logical steps that build on each other
3. For each step, provide:
   - A clear name (max 50 chars)
   - A detailed description of what the step does
   - Whether it requires user confirmation before execution
   - Whether it's critical (failure stops the workflow)
   - Any command or action required
   - Data dependencies on previous steps

Do not include steps that are too vague or cannot be executed programmatically.
Format your response as a JSON object with the following structure:

{
  "workflow_name": "Brief descriptive name",
  "workflow_description": "Detailed explanation of what this workflow will do",
  "steps": [
    {
      "name": "Step name",
      "description": "Detailed description of what this step does",
      "requires_confirmation": true/false,
      "critical": true/false,
      "action": "Command or action to execute",
      "dependencies": []
    },
    ...
  ]
}

Remember that each step should be specific, actionable, and contribute directly to the overall goal.
"""

STEP_REASONING_PROMPT = """
You are an expert AI workflow executor focusing on a single step in a multi-step workflow.

Current workflow: ${workflow_name}
Workflow description: ${workflow_description}
Current step: ${step_name}
Step description: ${step_description}

Previous steps and their results:
${previous_steps}

User request: ${user_request}
System context: ${system_context}

Your task is to reason through how to execute this step successfully.
Consider:
1. What specific actions are needed
2. How to handle potential errors
3. How to interpret and use outputs from previous steps
4. What success looks like for this step

Provide your reasoning and then the exact command or action to take.
Format your response as a JSON object with:
{
  "reasoning": "Your step-by-step reasoning process",
  "action": "The specific command or action to execute",
  "expected_outcome": "What you expect to happen upon successful execution"
}
"""

ERROR_RECOVERY_PROMPT = """
You are an expert AI workflow troubleshooter. A step in the workflow has failed, and you need to analyze the error and 
suggest recovery options.

Workflow: ${workflow_name}
Workflow description: ${workflow_description}
Failed step: ${step_name} 
Step description: ${step_description}
Error details: ${error_details}
Command that failed: ${command}
Command output: ${command_output}

Your task is to:
1. Analyze what went wrong
2. Identify potential root causes
3. Suggest 1-3 recovery strategies, which could include:
   - Adjusting the command and retrying
   - Suggesting an alternative approach
   - Recommending the workflow be terminated
   - Skipping this step if not critical

Format your response as a JSON object with:
{
  "error_analysis": "Your analysis of what went wrong",
  "root_causes": ["Potential cause 1", "Potential cause 2", ...],
  "recovery_strategies": [
    {
      "strategy": "Short strategy name",
      "description": "Detailed description of the recovery approach",
      "action": "Command or action to execute if this strategy is chosen",
      "confidence": 0-100
    },
    ...
  ],
  "recommended_strategy": "strategy_name"
}
"""


class WorkflowManager:
    """
    Manages the creation, execution, and monitoring of multi-step workflows.
    """
    
    def __init__(
        self,
        agent_core: AgentCore,
        context_manager: ContextManager,
        reasoning_engine: Optional[ReasoningEngine] = None,
        workflows_dir: Optional[str] = None,
        max_retries: int = 3,
        auto_save: bool = True,
    ):
        """
        Initialize the workflow manager.
        
        Args:
                    if recovery_result["status"] == "recovered":
                        # Recovery succeeded
                        step.mark_completed({
                            "command": action,
                            "original_output": result.output,
                            "recovery_action": recovery_result.get("action", ""),
                            "recovery_output": recovery_result.get("output", ""),
                        })
                        workflow.completed_steps.append(step.step_id)
                    elif recovery_result["status"] == "skip":
                        # Skip this step
                        step.mark_skipped(recovery_result.get("reason", "Skipped during recovery"))
                        workflow.completed_steps.append(step.step_id)
                    else:
                        # Recovery failed
                        if step.critical:
                            # Critical step failed, mark workflow as failed
                            step.mark_failed(recovery_result.get("error", "Recovery failed"))
                            workflow.failed_steps.append(step.step_id)
                            workflow.mark_failed(f"Critical step {step.name} failed")
                        else:
                            # Non-critical step failed, mark it and continue
                            step.mark_failed(recovery_result.get("error", "Recovery failed"))
                            workflow.failed_steps.append(step.step_id)
                else:
                    # Command succeeded
                    step.mark_completed({
                        "command": action,
                        "output": result.output,
                        "exit_code": result.exit_code,
                    })
                    workflow.completed_steps.append(step.step_id)
            else:
                # It's a callable function
                try:
                    # Execute the function
                    function_result = action(step.params)
                    step.mark_completed({
                        "function": str(action),
                        "result": str(function_result),
                    })
                    workflow.completed_steps.append(step.step_id)
                except Exception as e:
                    error_msg = f"Function execution error: {str(e)}"
                    logger.error(error_msg)
                    
                    if step.critical:
                        step.mark_failed(error_msg)
                        workflow.failed_steps.append(step.step_id)
                        workflow.mark_failed(f"Critical step {step.name} failed")
                    else:
                        step.mark_failed(error_msg)
                        workflow.failed_steps.append(step.step_id)
            
            # Return execution result
            return {
                "status": step.status.value,
                "step_id": step.step_id,
                "name": step.name,
                "result": step.result,
            }
            
        except Exception as e:
            error_msg = f"Step execution error: {str(e)}"
            logger.error(error_msg)
            step.mark_failed(error_msg)
            workflow.failed_steps.append(step.step_id)
            
            if step.critical:
                workflow.mark_failed(f"Critical step {step.name} failed: {str(e)}")
            
            return {
                "status": "failed",
                "step_id": step.step_id,
                "name": step.name,
                "error": str(e),
            }
            max_retries: Maximum retry attempts for failed steps
            auto_save: Whether to automatically save workflow state
        """
        self.agent_core = agent_core
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine or ReasoningEngine(
            context_manager=context_manager
        )
        
        # Prepare result summary
        result_summary = {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "steps_total": len(workflow.steps),
            "steps_completed": len(workflow.completed_steps),
            "steps_failed": len(workflow.failed_steps),
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "error": workflow.error,
            "execution_log": workflow.execution_log,
        }
        
        logger.info("WorkflowManager initialized")
    
    async def _get_llm_response(
        self, 
        system_message: str, 
        user_message: str, 
        validation_types: List[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Get a response from the LLM.
        
        Args:
            system_message: System message for the LLM
            user_message: User message for the LLM
            validation_types: Types of validation to perform
            schema: JSON schema for validation
            max_retries: Maximum number of retry attempts
            
        Returns:
            The LLM response
        """
        # This is a placeholder implementation
        # In a real system, this would use the OpenAI wrapper or similar
        try:
            # Use the agent core's LLM provider
            response = await self.agent_core.get_llm_response(
                system_message=system_message,
                user_message=user_message,
            )
            
            # Validate the response if needed
            if validation_types:
                is_valid, error = self.reasoning_engine.validator.validate(
                    response=response,
                    validation_types=validation_types,
                    schema=schema,
                )
                
                if not is_valid:
                    raise ResponseValidationError(error or "Response validation failed")
            
            return response
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            raise
    
    async def _execute_command(self, command: str) -> CommandResult:
        """
        Execute a shell command.
        
        Args:
            command: The command to execute
            
        Returns:
            Command execution result
        """
        # Use the agent core to execute the command
        return await self.agent_core.execute_command(command)
    
    async def _reason_about_step(self, step: WorkflowStep, workflow: Workflow) -> Dict[str, Any]:
        """
        Use LLM to reason about how to execute a workflow step.
        
        Args:
            step: The step to reason about
            workflow: The parent workflow
            
        Returns:
            Reasoning result with action and expected outcome
        """
        # Get previous steps and their results for context
        previous_steps_text = ""
        for prev_step_id in workflow.completed_steps:
            prev_step = workflow.get_step(prev_step_id)
            if prev_step:
                step_result = prev_step.result
                result_desc = json.dumps(step_result) if step_result else "No result"
                previous_steps_text += f"Step: {prev_step.name}\nDescription: {prev_step.description}\nResult: {result_desc}\n\n"
        
        if not previous_steps_text:
            previous_steps_text = "No previous steps completed yet."
        
        # Get system context
        system_context = self._get_system_context()
        
        # Format the prompt
        system_message, user_message = self.reasoning_engine.templates["step_reasoning"].format({
            "workflow_name": workflow.name,
            "workflow_description": workflow.description,
            "step_name": step.name,
            "step_description": step.description,
            "previous_steps": previous_steps_text,
            "user_request": workflow.user_request,
            "system_context": system_context,
        })
        
        # Get reasoning from LLM
        try:
            reasoning_response = await self._get_llm_response(
                system_message=system_message,
                user_message=user_message,
                validation_types=["json_schema"],
            )
            
            # Parse response
            reasoning_data = json.loads(reasoning_response)
            return reasoning_data
        except Exception as e:
            logger.warning(f"Error in step reasoning, using original action: {str(e)}")
            return {
                "reasoning": "Failed to generate reasoning",
                "action": step.action,
                "expected_outcome": "Original step execution",
            }
    
    async def _handle_step_error(
        self, 
        step: WorkflowStep, 
        workflow: Workflow, 
        command: str, 
        output: str, 
        error: str
    ) -> Dict[str, Any]:
        """
        Handle a step execution error using LLM-based recovery.
        
        Args:
            step: The failed step
            workflow: The parent workflow
            command: The command that failed
            output: Command output
            error: Error message
            
        Returns:
            Recovery result
        """
        # Format the prompt for error recovery
        system_message, user_message = self.reasoning_engine.templates["error_recovery"].format({
            "workflow_name": workflow.name,
            "workflow_description": workflow.description,
            "step_name": step.name,
            "step_description": step.description,
            "error_details": error,
            "command": command,
            "command_output": output,
        })
        
        # Get recovery strategies from LLM
        try:
            recovery_response = await self._get_llm_response(
                system_message=system_message,
                user_message=user_message,
                validation_types=["json_schema"],
            )
            
            # Parse response
            recovery_data = json.loads(recovery_response)
            
            # Get recommended strategy
            recommended_strategy = recovery_data.get("recommended_strategy")
            if not recommended_strategy:
                # If no recommendation, pick the highest confidence strategy
                strategies = recovery_data.get("recovery_strategies", [])
                if strategies:
                    strategies.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                    recommended_strategy = strategies[0]["strategy"]
            
            # Find the recommended strategy details
            for strategy in recovery_data.get("recovery_strategies", []):
                if strategy["strategy"] == recommended_strategy:
                    action = strategy.get("action", "")
                    
                    if recommended_strategy.lower() == "retry":
                        # Retry with the same or modified command
                        retry_action = action or command
                        result = await self._execute_command(retry_action)
                        
                        if result.exit_code == 0:
                            return {
                                "status": "recovered",
                                "action": retry_action,
                                "output": result.output,
                            }
                        else:
                            return {
                                "status": "failed",
                                "error": "Retry failed",
                                "action": retry_action,
                                "output": result.output,
                            }
                    
                    elif recommended_strategy.lower() == "skip":
                        # Skip this step
                        return {
                            "status": "skip",
                            "reason": strategy.get("description", "Step skipped as recovery strategy"),
                        }
                    
                    elif recommended_strategy.lower() == "terminate":
                        # Terminate the workflow
                        return {
                            "status": "failed",
                            "error": strategy.get("description", "Workflow termination recommended"),
                        }
                    
                    elif recommended_strategy.lower() == "alternative":
                        # Try alternative approach
                        if action:
                            result = await self._execute_command(action)
                            
                            if result.exit_code == 0:
                                return {
                                    "status": "recovered",
                                    "action": action,
                                    "output": result.output,
                                }
                            else:
                                return {
                                    "status": "failed",
                                    "error": "Alternative approach failed",
                                    "action": action,
                                    "output": result.output,
                                }
                        else:
                            return {
                                "status": "failed",
                                "error": "No alternative action provided",
                            }
            
            # If we get here, no valid strategy was found
            if step.retry_count < step.max_retries:
                # Increment retry count and retry
                step.retry_count += 1
                return {
                    "status": "retry",
                    "retry_count": step.retry_count,
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Maximum retries ({step.max_retries}) exceeded",
                }
                
        except Exception as e:
            logger.error(
        templates = [
            PromptTemplate(
                template_id="workflow_planning",
                system_message="""You are an expert AI workflow planner that breaks down complex tasks into executable steps.""",
                user_message=WORKFLOW_PLANNING_PROMPT,
                variables=["user_request", "system_context"],
                metadata={"purpose": "Plan multi-step workflows"},
            ),
            PromptTemplate(
                template_id="step_reasoning",
                system_message="""You are an expert AI workflow executor focusing on a single step in a multi-step workflow.""",
                user_message=STEP_REASONING_PROMPT,
                variables=["workflow_name", "workflow_description", "step_name", 
                          "step_description", "previous_steps", "user_request", "system_context"],
                metadata={"purpose": "Reason through workflow step execution"},
            ),
            PromptTemplate(
                template_id="error_recovery",
                system_message="""You are an expert AI workflow troubleshooter that analyzes errors and suggests recovery options.""",
                user_message=ERROR_RECOVERY_PROMPT,
                variables=["workflow_name", "workflow_description", "step_name", 
                          "step_description", "error_details", "command", "command_output"],
                metadata={"purpose": "Recover from workflow execution errors"},
            ),
        ]
        
        for template in templates:
            self.reasoning_engine.templates[template.template_id] = template
            
        logger.info(f"Registered {len(templates)} workflow templates")
    
    async def confirm_step(self, workflow_id: str, step_id: str, confirmed: bool) -> Dict[str, Any]:
        """
        Confirm or reject a step that is waiting for user confirmation.
        
        Args:
            workflow_id: The ID of the workflow
            step_id: The ID of the step to confirm
            confirmed: Whether the step is confirmed or rejected
            
        Returns:
            Result of the confirmation
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.active_workflows[workflow_id]
        step = workflow.get_step(step_id)
        
        if not step:
            raise ValueError(f"Step {step_id} not found in workflow {workflow_id}")
            
        if step.status != StepStatus.WAITING_CONFIRMATION:
            raise ValueError(f"Step {step_id} is not waiting for confirmation")
            
        if confirmed:
            # Resume workflow execution
            workflow.status = WorkflowStatus.IN_PROGRESS
            result = await self._execute_step(step, workflow)
            
            # Save workflow state
            if self.auto_save:
                self._save_workflow(workflow)
                
            return {
                "status": "confirmed",
                "step_id": step_id,
                "result": result,
            }
        else:
            # Mark step as cancelled
            step.mark_cancelled("Rejected by user")
            
            # If step is critical, cancel the workflow
            if step.critical:
                workflow.mark_failed("Critical step rejected by user")
            
            # Save workflow state
            if self.auto_save:
                self._save_workflow(workflow)
                
            return {
                "status": "rejected",
                "step_id": step_id,
                "workflow_status": workflow.status.value,
            }
    
    async def execute_workflow(
        self, workflow_id: str, auto_confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new workflow based on a user request.
        
        Args:
            user_request: The user's natural language request
            user_id: Optional user identifier
            
        Returns:
            A new Workflow instance with planned steps
        """
        # Generate a unique workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Get system context for planning
        system_context = self._get_system_context()
        
        # Use LLM to plan workflow steps
        try:
            # Format the prompt with user request and system context
            system_message, user_message = self.reasoning_engine.templates["workflow_planning"].format({
                "user_request": user_request,
                "system_context": system_context,
            })
            
            # Get plan from LLM
            plan_response = await self._get_llm_response(
                system_message=system_message,
                user_message=user_message,
                validation_types=["json_schema"],
            )
            
            # Parse plan
            plan_data = json.loads(plan_response)
            
            # Create the workflow object
            workflow = Workflow(
                workflow_id=workflow_id,
                name=plan_data["workflow_name"],
                description=plan_data["workflow_description"],
                user_request=user_request,
                context_manager=self.context_manager,
            )
            
            # Add steps to the workflow
            for i, step_data in enumerate(plan_data["steps"]):
                step = WorkflowStep(
                    step_id=f"{workflow_id}_step_{i+1}",
                    name=step_data["name"],
                    description=step_data["description"],
                    action=step_data["action"],
                    workflow_id=workflow_id,
                    requires_confirmation=step_data.get("requires_confirmation", False),
                    critical=step_data.get("critical", False),
                    dependencies=step_data.get("dependencies", []),
                )
                workflow.add_step(step)
            
            # Store the workflow
            self.active_workflows[workflow_id] = workflow
            
            # Save workflow state if auto_save is enabled
            if self.auto_save:
                self._save_workflow(workflow)
                
            logger.info(f"Created workflow {workflow_id} with {len(workflow.steps)} steps")
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            raise ValueError(f"Failed to create workflow: {str(e)}")
    
    async def execute_workflow(
        self, workflow_id: str, auto_confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a workflow by running all its steps in sequence.
        
        Args:
            workflow_id: The ID of the workflow to execute
            auto_confirm: Whether to automatically confirm all steps
            
        Returns:
            Execution result summary
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.active_workflows[workflow_id]
        
        # Mark workflow as started
        workflow.mark_started()
        logger.info(f"Starting execution of workflow {workflow_id}")
        
        # Execute steps in sequence, respecting dependencies
        while workflow.status == WorkflowStatus.IN_PROGRESS:
            # Get next executable steps
            next_steps = workflow.get_next_executable_steps()
            
            if not next_steps:
                # Check if all steps are completed
                all_steps_completed = all(
                    step.status in [StepStatus.SUCCEEDED, StepStatus.SKIPPED]
                    for step in workflow.steps
                )
                
                if all_steps_completed:
                    workflow.mark_completed()
                    logger.info(f"Workflow {workflow_id} completed successfully")
                else:
                    # Check for failed steps
                    failed_steps = [
                        step for step in workflow.steps if step.status == StepStatus.FAILED
                    ]
                    
                    if failed_steps:
                        workflow.mark_failed(
                            f"Workflow failed due to {len(failed_steps)} failed steps"
                        )
                        logger.error(f"Workflow {workflow_id} failed")
                    else:
                        # If there are no executable steps but not all are completed,
                        # there might be a dependency cycle
                        workflow.mark_failed("Workflow stalled due to dependency issues")
                        logger.error(f"Workflow {workflow_id} stalled")
                
                break
            
            # Execute each ready step
            for step in next_steps:
                result = await self._execute_step(step, workflow, auto_confirm)
                
                # Update workflow status if needed
                if workflow.status != WorkflowStatus.IN_PROGRESS:
                    break
            
            # Save workflow state after each step execution
            if self.auto_save:
                self._save_workflow(workflow)
        
        # Prepare result summary
        result_summary = {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "steps_total": len(workflow.steps),
            "steps_completed": len(workflow.completed_steps),
            "steps_failed": len(workflow.failed_steps),
            "started_at": workflow.started_at,
            "completed_at": workflow.completed_at,
            "error": workflow.error,
        }
        
        return result_summary
    
    async def _execute_step(
        self, step: WorkflowStep, workflow: Workflow, auto_confirm: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single workflow step.
        
        Args:
            step: The step to execute
            workflow: The parent workflow
            auto_confirm: Whether to auto-confirm this step
            
        Returns:
            Step execution result
        """
        logger.info(f"Executing step {step.step_id}: {step.name}")
        
        # Update the current step
        workflow.current_step_id = step.step_id
        
        # Check if user confirmation is required
        if step.requires_confirmation and not auto_confirm:
            step.mark_waiting_confirmation()
            workflow.status = WorkflowStatus.WAITING_USER_INPUT
            
            # This would typically prompt the user in a real system
            # For now, we'll just log it
            logger.info(f"Step {step.step_id} is waiting for user confirmation")
            
            # Return early, execution will continue after confirmation
            return {"status": "waiting_confirmation"}
        
        # Mark step as started
        step.mark_started()
        
        try:
            # Use LLM to reason about step execution
            reasoning_result = await self._reason_about_step(step, workflow)
            
            # Execute the action
            action = reasoning_result.get("action", step.action)
            expected_outcome = reasoning_result.get("expected_outcome", "Successful execution")
            
            # Execute the command
            if isinstance(action, str):
                # It's a command string
                result = await self._execute_command(action)
                
                if result.exit_code != 0:
                    # Command failed, attempt recovery
                    recovery_result = await self._handle_step_error(
                        step, workflow, action, result.output, result.error or "Non-zero exit code"
                    )
                    
                    if recovery_result["status"] == "recovered":
                        # Recovery succeeded
                        step.mark_completed({
                            "command": action,

