"""
Auto Detector module for environment and preference detection.

This module provides an AutoDetector class that automatically detects
shell settings, OS details, user preferences, and project types.
"""

import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            Path(__file__).parent.parent / "data" / "logs" / "auto_detector.log"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("auto_detector")


class AutoDetector:
    """
    Detects environment settings, user preferences, project types,
    and terminal capabilities automatically.
    """

    def __init__(self, context_manager=None):
        """
        Initialize the Auto Detector.

        Args:
            context_manager: Component for tracking context
        """
        self.context_manager = context_manager
        
        # Detection results
        self.environment_info: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.project_info: Dict[str, Any] = {}
        self.terminal_capabilities: Dict[str, Any] = {}
        
        # Cached command output for performance
        self._command_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("AutoDetector initialized")

    def detect_all(self) -> Dict[str, Any]:
        """
        Perform all available detection methods.

        Returns:
            Dictionary with all detected information
        """
        try:
            # Detect environment
            self.detect_environment()
            
            # Detect user preferences
            self.detect_user_preferences()
            
            # Detect project type
            self.detect_project_type()
            
            # Detect terminal capabilities
            self.detect_terminal_capabilities()
            
            # Combine all detection results
            all_info = {
                "environment": self.environment_info,
                "user_preferences": self.user_preferences,
                "project": self.project_info,
                "terminal": self.terminal_capabilities,
            }
            
            # Store in context manager if available
            if self.context_manager:
                self.context_manager.set_user_variable("detected_environment", all_info)
                
            return all_info
            
        except Exception as e:
            logger.error(f"Error in detect_all: {str(e)}")
            return {
                "environment": self.environment_info,
                "user_preferences": self.user_preferences,
                "project": self.project_info,
                "terminal": self.terminal_capabilities,
                "error": str(e),
            }

    def detect_environment(self) -> Dict[str, Any]:
        """
        Detect operating system, shell, and environment settings.

        Returns:
            Dictionary with environment information
        """
        try:
            # Detect OS
            os_info = self._detect_os_info()
            self.environment_info["os"] = os_info
            
            # Detect shell
            shell_info = self._detect_shell_info()
            self.environment_info["shell"] = shell_info
            
            # Detect Python
            python_info = self._detect_python_info()
            self.environment_info["python"] = python_info
            
            # Detect environment variables
            env_vars = self._detect_environment_variables()
            self.environment_info["env_vars"] = env_vars
            
            # Detect network info
            network_info = self._detect_network_info()
            self.environment_info["network"] = network_info
            
            logger.info(f"Detected environment info: {list(self.environment_info.keys())}")
            return self.environment_info
            
        except Exception as e:
            logger.error(f"Error detecting environment: {str(e)}")
            return self.environment_info

    def detect_user_preferences(self) -> Dict[str, Any]:
        """
        Detect user preferences from dotfiles and configuration files.

        Returns:
            Dictionary with user preference information
        """
        try:
            # Detect editor preferences
            editor_prefs = self._detect_editor_preferences()
            self.user_preferences["editor"] = editor_prefs
            
            # Detect shell preferences
            shell_prefs = self._detect_shell_preferences()
            self.user_preferences["shell"] = shell_prefs
            
            # Detect Git preferences
            git_prefs = self._detect_git_preferences()
            self.user_preferences["git"] = git_prefs
            
            # Detect theme preferences
            theme_prefs = self._detect_theme_preferences()
            self.user_preferences["theme"] = theme_prefs
            
            logger.info(f"Detected user preferences: {list(self.user_preferences.keys())}")
            return self.user_preferences
            
        except Exception as e:
            logger.error(f"Error detecting user preferences: {str(e)}")
            return self.user_preferences

    def detect_project_type(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect the type of project in the current directory.

        Args:
            directory: Directory to analyze (default: current directory)

        Returns:
            Dictionary with project information
        """
        try:
            dir_path = directory or os.getcwd()
            
            # Initialize project info
            self.project_info = {
                "directory": dir_path,
                "name": os.path.basename(dir_path),
                "type": [],
                "detected_files": {},
            }
            
            # Check for version control
            vc_info = self._detect_version_control(dir_path)
            self.project_info["version_control"] = vc_info
            
            # Check for package managers
            package_managers = self._detect_package_managers(dir_path)
            self.project_info["package_managers"] = package_managers
            
            # Check for project types
            project_types = []
            
            # JavaScript/Node.js project
            if os.path.exists(os.path.join(dir_path, "package.json")):
                project_types.append("nodejs")
                self.project_info["detected_files"]["package.json"] = True
                
            # Python project
            if any(os.path.exists(os.path.join(dir_path, f)) for f in ["setup.py", "pyproject.toml", "requirements.txt"]):
                project_types.append("python")
                
                # Specific Python project type
                if os.path.exists(os.path.join(dir_path, "pyproject.toml")):
                    self.project_info["detected_files"]["pyproject.toml"] = True
                    # Check if it's Poetry
                    with open(os.path.join(dir_path, "pyproject.toml"), "r") as f:
                        content = f.read()
                        if "[tool.poetry]" in content:
                            project_types.append("poetry")
                if os.path.exists(os.path.join(dir_path, "setup.py")):
                    self.project_info["detected_files"]["setup.py"] = True
                if os.path.exists(os.path.join(dir_path, "requirements.txt")):
                    self.project_info["detected_files"]["requirements.txt"] = True
                    
            # Ruby project
            if os.path.exists(os.path.join(dir_path, "Gemfile")):
                project_types.append("ruby")
                self.project_info["detected_files"]["Gemfile"] = True
                
            # Rust project
            if os.path.exists(os.path.join(dir_path, "Cargo.toml")):
                project_types.append("rust")
                self.project_info["detected_files"]["Cargo.toml"] = True
                
            # Go project
            if os.path.exists(os.path.join(dir_path, "go.mod")):
                project_types.append("go")
                self.project_info["detected_files"]["go.mod"] = True
                
            # Java project
            if any(os.path.exists(os.path.join(dir_path, f)) for f in ["pom.xml", "build.gradle"]):
                project_types.append("java")
                if os.path.exists(os.path.join(dir_path, "pom.xml")):
                    self.project_info["detected_files"]["pom.xml"] = True
                    project_types.append("maven")
                if os.path.exists(os.path.join(dir_path, "build.gradle")):
                    self.project_info["detected_files"]["build.gradle"] = True
                    project_types.append("gradle")
                    
            # Docker project
            if os.path.exists(os.path.join(dir_path, "Dockerfile")) or os.path.exists(os.path.join(dir_path, "docker-compose.yml")):
                project_types.append("docker")
                if os.path.exists(os.path.join(dir_path, "Dockerfile")):
                    self.project_info["detected_files"]["Dockerfile"] = True
                if os.path.exists(os.path.join(dir_path, "docker-compose.yml")):
                    self.project_info["detected_files"]["docker-compose.yml"] = True
                    
            # Set detected project types
            self.project_info["type"] = project_types
            
            # Get directory structure
            self.project_info["structure"] = self._get_directory_structure(dir_path)
            
            logger.info(f"Detected project types: {project_types}")
            return self.project_info
            
        except Exception as e:
            logger.error(f"Error detecting project type: {str(e)}")
            return self.project_info

    def detect_terminal_capabilities(self) -> Dict[str, Any]:
        """
        Detect terminal capabilities like color support, size, etc.

        Returns:
            Dictionary with terminal capability information
        """
        try:
            # Detect terminal size
            term_size = self._detect_terminal_size()
            self.terminal_capabilities["size"] = term_size
            
            # Detect color support
            color_support = self._detect_color_support()
            self.terminal_capabilities["color"] = color_support
            
            # Detect terminal type
            term_type = self._detect_terminal_type()
            self.terminal_capabilities["type"] = term_type
            
            # Unicode support
            unicode_support = self._detect_unicode_support()
            self.terminal_capabilities["unicode"] = unicode_support
            
            logger.info(f"Detected terminal capabilities: {list(self.terminal_capabilities.keys())}")
            return self.terminal_capabilities
            
        except Exception as e:
            logger.error(f"Error detecting terminal capabilities: {str(e)}")
            return self.terminal_capabilities

    def _run_command(self, command: str, shell: bool = True) -> Dict[str, Any]:
        """
        Run a shell command and return its output with caching.

        Args:
            command: Command to execute
            shell: Whether to use shell execution

        Returns:
            Dictionary with command result
        """
        # Check cache first
        if command in self._command_cache:
            return self._command_cache[command]
            
        try:
            # Run the command
            process = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=5  # Timeout to prevent hanging
            )
            
            result = {
                "output": process.stdout.strip(),
                "error": process.stderr.strip(),
                "exit_code": process.returncode,
                "success": process.returncode == 0,
            }
            
            # Cache the result
            self._command_cache[command] = result
            return result
            
        except subprocess.TimeoutExpired:
            result = {
                "output": "",
                "error": f"Command timed out: {command}",
                "exit_code": 124,
                "success": False,
            }
            self._command_cache[command] = result
            return result
            
        except Exception as e:
            result = {
                "output": "",
                "error": str(e),
                "exit_code": 1,
                "success": False,
            }
            self._command_cache[command] = result
            return result

    def _detect_os_info(self) -> Dict[str, Any]:
        """Detect operating system information."""
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Specific OS detection
        if os_info["system"] == "Darwin":
            # macOS
            os_info["type"] = "macOS"
            
            # Get macOS version
            mac_ver = platform.mac_ver()
            os_info["mac_version"] = mac_ver[0]
            
            # Try to get macOS name (Big Sur, Monterey, etc.)
            try:
                sw_vers = self._run_command("sw_vers -productVersion")
                if sw_vers["success"]:
                    os_info["macos_version"] = sw_vers["output"]
            except Exception:
                pass
                
        elif os_info["system"] == "Linux":
            # Linux
            os_info["type"] = "Linux"
            
            # Try to detect Linux distribution
            try:
                # Try lsb_release first
                if shutil.which("lsb_release"):
                    lsb = self._run_command("lsb_release -ds")
                    if lsb["success"]:
                        os_info["distribution"] = lsb["output"].strip('"')
                
                # If lsb_release failed, check /etc/os-release
                if "distribution" not in os_info and os.path.exists("/etc/os-release"):
                    with open("/etc/os-release", "r") as f:
                        for line in f:
                            if line.startswith("PRETTY_NAME="):
                                os_info["distribution"] = line.split("=")[1].strip().strip('"')
                                break
                                
            except Exception:
                os_info["distribution"] = "Unknown"
                
        elif os_info["system"] == "Windows":
            # Windows
            os_info["type"] = "Windows"
            
            # Get Windows version details
            win_ver = platform.win32_ver()
            os_info["windows_version"] = win_ver[0]
            os_info["windows_edition"] = win_ver[1]
            
        return os_info

    def _detect_shell_info(self) -> Dict[str, Any]:
        """Detect shell information."""
        shell_info = {
            "name": "unknown",
            "version": "unknown",
            "path": os.environ.get("SHELL", ""),
        }
        
        # Get shell from environment
        if shell_info["path"]:
            shell_name = os.path.basename(shell

