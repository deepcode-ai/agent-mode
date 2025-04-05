"""
Agent Mode core package.

This package provides the core components for the Agent Mode system,
enabling AI-assisted command-line interactions.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

# Version information
__version__ = "0.1.0"
__author__ = "Agent Mode Team"
__license__ = "MIT"
__copyright__ = "Copyright 2025"

# Set up logging
log_dir = Path(__file__).parent.parent / "data" / "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "agent.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("agent")

# Import core components
try:
    from .context_manager import ContextManager
    from .reasoning import ReasoningEngine, PromptTemplate, ResponseValidator
    from .planner import TaskPlanner, PlanStep, TaskPlan, PlanStatus, StepStatus
    
    # Note: These imports should be added as they are implemented
    # from .permission_manager import PermissionManager
    # from .cli_executor import CLIExecutor
    # from .plugin_manager import PluginManager
    
    logger.info(f"Agent Mode Core v{__version__} initialized")
except ImportError as e:
    logger.error(f"Error importing core components: {e}")
    raise

# Define the component type for dependency injection
T = TypeVar('T')

# Global component registry
_components: Dict[str, Any] = {}


def register_component(component_type: Type[T], instance: T) -> None:
    """
    Register a component in the global registry.
    
    Args:
        component_type: The type of the component
        instance: The component instance
    """
    _components[component_type.__name__] = instance
    logger.debug(f"Registered component: {component_type.__name__}")


def get_component(component_type: Type[T]) -> Optional[T]:
    """
    Get a component from the global registry.
    
    Args:
        component_type: The type of the component to retrieve
        
    Returns:
        The component instance or None if not found
    """
    return _components.get(component_type.__name__)


def initialize_components() -> Dict[str, Any]:
    """
    Initialize all core components with proper dependency injection.
    
    Returns:
        Dictionary of initialized components
    """
    logger.info("Initializing components...")
    
    # Create component instances in dependency order
    try:
        # First, create context manager as it has no dependencies
        context_manager = ContextManager()
        register_component(ContextManager, context_manager)
        
        # Then create CLI executor
        try:
            from .cli_executor import CLIExecutor
            cli_executor = CLIExecutor(context_manager=context_manager)
            register_component(CLIExecutor, cli_executor)
        except ImportError:
            logger.warning("CLIExecutor not available")
            cli_executor = None
        
        # Create permission manager
        try:
            from .permission_manager import PermissionManager
            permission_manager = PermissionManager(context_manager=context_manager)
            register_component(PermissionManager, permission_manager)
        except ImportError:
            logger.warning("PermissionManager not available")
            permission_manager = None
        
        # Create reasoning engine
        reasoning_engine = ReasoningEngine(context_manager=context_manager)
        register_component(ReasoningEngine, reasoning_engine)
        
        # Create task planner
        task_planner = TaskPlanner(
            cli_executor=cli_executor,
            permission_manager=permission_manager,
            context_manager=context_manager
        )
        register_component(TaskPlanner, task_planner)
        
        # Create plugin manager
        try:
            from .plugin_manager import PluginManager
            plugin_manager = PluginManager(
                cli_executor=cli_executor,
                context_manager=context_manager
            )
            register_component(PluginManager, plugin_manager)
        except ImportError:
            logger.warning("PluginManager not available")
            plugin_manager = None
        
        # Initialize interface components
        try:
            from ..interface.auto_detector import AutoDetector
            auto_detector = AutoDetector(context_manager=context_manager)
            register_component(AutoDetector, auto_detector)
            
            # Run detection
            auto_detector.detect_all()
        except ImportError:
            logger.warning("AutoDetector not available")
            auto_detector = None
        
        try:
            from ..interface.output_formatter import OutputFormatter
            output_formatter = OutputFormatter(context_manager=context_manager)
            register_component(OutputFormatter, output_formatter)
        except ImportError:
            logger.warning("OutputFormatter not available")
            output_formatter = None
        
        logger.info("Component initialization complete")
        
        return _components
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise


# Constants
DEFAULT_CONFIG_PATH = Path.home() / ".agent-mode" / "config.json"
DEFAULT_DATA_DIR = Path.home() / ".agent-mode" / "data"
DEFAULT_LOGS_DIR = Path.home() / ".agent-mode" / "logs"

# Package-level utilities
def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return os.environ.get("AGENT_MODE_DEBUG", "").lower() in ("1", "true", "yes")


def get_version_info() -> Dict[str, str]:
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "copyright": __copyright__,
        "python": sys.version,
    }

