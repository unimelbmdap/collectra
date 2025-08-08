from __future__ import annotations
import os
from shutil import rmtree
import shutil
from typing import List, Optional, Union
from pathlib import Path
from rich import print
from rocrate.rocrate import ROCrate
from .utils import BaseROCrate
from .service import TaskManager
from .task import TaskEntity


class WorkflowConfig:
    """Configuration class for workflow settings.
    
    This class encapsulates all configuration parameters needed for a workflow,
    providing validation and centralized access to settings.
    """

    def __init__(
        self, name: str, version: str, output: Path, file_format: Optional[str] = None
    ):
        # Core workflow identification
        self.name = name  # Unique name for the workflow
        self.version = version  # Version string for workflow versioning
        self.output = Path(output)  # Output directory for workflow artifacts
        
        # File format configuration with sensible default
        self.file_format = file_format or "grapto"  # Default to grapto format
        
        # Temporary directory for intermediate processing files
        self.temporary_dir = Path("tmp")  # Standard temp directory location

    def validate(self) -> bool:
        """Validate configuration parameters.
        
        Ensures all required configuration values are present and valid.
        Raises ValueError for invalid configurations.
        
        Returns:
            bool: True if all validations pass
            
        Raises:
            ValueError: If name or version is empty/invalid
        """
        # Validate workflow name is not empty or None
        if not self.name:
            raise ValueError("Workflow name cannot be empty")
            
        # Validate version string is present
        if not self.version:
            raise ValueError("Workflow version cannot be empty")
            
        return True  # All validations passed


class WorkflowFileManager:
    """Handles file operations for workflows.
    
    This class manages all file system operations including saving workflows,
    managing temporary files, and cleanup operations. It encapsulates file
    handling logic to keep the main workflow class focused on business logic.
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize file manager with workflow configuration.
        
        Args:
            config: WorkflowConfig instance containing file paths and settings
        """
        self.config = config  # Store reference to workflow configuration

    def save_workflow(self, crate: ROCrate, compressed: bool = True) -> None:
        """Save workflow to file system.
        
        Saves the ROCrate workflow to disk, either as a compressed ZIP file
        or as an uncompressed directory structure.
        
        Args:
            crate: ROCrate instance containing the workflow data
            compressed: If True, save as ZIP file; if False, save as directory
        """
        # Construct the full file path for the workflow
        file_path = self.config.output / self.config.name

        if compressed:
            # Handle compressed (ZIP) format
            if file_path.exists():
                # Warn user about file replacement and remove existing file
                print(
                    f"[bold red]Replacing existing workflow file[/bold red]: {file_path}"
                )
                os.remove(file_path)
            # Write workflow as compressed ZIP archive
            crate.write_zip(file_path)
        else:
            # Write workflow as uncompressed directory structure
            crate.write(file_path)
        
        run_folder = Path.cwd() / "runs" 
        yolo_pt_file = Path.cwd() / "yolo11n.pt"

        if run_folder.exists():
            shutil.rmtree(run_folder)
        if yolo_pt_file.exists():        
            os.remove(yolo_pt_file)

    def cleanup_temporary_files(self, tasks: List[TaskEntity] = None) -> None:
        """Clean up temporary files and directories.
        
        Removes temporary files created during workflow execution, including
        task-specific engine files and the main temporary directory.
        
        Args:
            tasks: Optional list of TaskEntity instances to clean up
        """
        print(f"[blue]Cleaning up temporary files...[/blue]")

        # Clean up task-specific temporary files
        if tasks:
            for task in tasks:
                # Check if task has an associated engine with temporary files
                if hasattr(task, "engine") and task.engine:
                    model_path = Path(task.engine.name)
                    # Remove temporary engine model files
                    if model_path.exists():
                        os.remove(model_path)
                        print(
                            f"[green]Removed temporary engine file[/green]: {model_path}"
                        )

        # Clean up the main temporary directory and all its contents
        if self.config.temporary_dir.exists():
            rmtree(self.config.temporary_dir, ignore_errors=True)
            print(
                f"[green]Removed temporary directory[/green]: {self.config.temporary_dir}"
            )


class WorkflowExecutor:
    """Handles workflow execution logic.
    
    This class manages the execution of workflows, including both inference
    (run) and training operations. It coordinates with TaskManager to execute
    individual tasks within the workflow.
    """

    def __init__(self, config: WorkflowConfig):
        """Initialize executor with workflow configuration.
        
        Args:
            config: WorkflowConfig instance containing execution settings
        """
        self.config = config  # Store configuration for execution context

    def run_workflow(self, crate: ROCrate, input_path: Path, output_path: Path) -> None:
        """Execute workflow with given input and output paths.
        
        Runs the complete workflow by chaining all tasks in the ROCrate
        and executing them in sequence with the provided input/output paths.
        
        Args:
            crate: ROCrate containing the workflow definition and tasks
            input_path: Path to input data for the workflow
            output_path: Path where workflow results will be saved
        """
        # Prepare execution configuration with paths and crate reference
        execution_config = {
            "input": input_path,      # Input data location
            "output": output_path,    # Output destination
            "crate": crate,           # Workflow definition
        }

        # Initialize task manager for workflow execution
        manager = TaskManager()
        
        # Iterate through all entities in the crate to find tasks
        for entity in crate.get_entities():
            if entity.type == "Task":
                # Chain each task with the execution configuration
                manager.chain(entity, execution_config)
                
        # Execute all chained tasks in sequence
        manager.run()

    def train_workflow(
        self,
        crate: ROCrate,
        task: str,
        input_files: List[str],
        output_path: Path,
        **kwargs,
    ) -> None:
        """Train workflow with specified parameters.
        
        Executes the training process for a specific task within the workflow.
        This involves setting up training configuration and delegating to
        TaskManager for the actual training execution.
        
        Args:
            crate: ROCrate containing the workflow and task definitions
            task: String identifier for the specific task to train
            input_files: List of file paths containing training data
            output_path: Path where training results will be saved
            **kwargs: Additional training parameters (e.g., test mode)
        """
        # Prepare comprehensive training configuration
        training_config = {
            "input": input_files,                    # Training data files
            "output": output_path,                   # Training output location
            "crate": crate,                          # Workflow definition
            "tmp_dir": self.config.temporary_dir,    # Temporary files location
            "test": kwargs.get("test", False),       # Test mode flag
            "task": str(task),                       # Task identifier
        }
        
        # Delegate training execution to TaskManager
        TaskManager.train(task, training_config)

    def eval_workflow(
        self,
        crate: ROCrate,
        task: str,
        input_files: List[str],
        output_path: Path,
        **kwargs,
    ) -> None:
        """Evaluate workflow with specified parameters.
        
        Executes the evaluation process for a specific task within the workflow.
        This involves setting up evaluation configuration and delegating to
        TaskManager for the actual evaluation execution.
        
        Args:
            crate: ROCrate containing the workflow and task definitions
            task: String identifier for the specific task to evaluate
            input_files: List of file paths containing evaluation data
            output_path: Path where evaluation results will be saved
            **kwargs: Additional evaluation parameters (e.g., test mode)
        """
        # Prepare comprehensive evaluation configuration
        evaluation_config = {
            "input": input_files,                    # Evaluation data files
            "output": output_path,                   # Evaluation output location
            "crate": crate,                          # Workflow definition
            "tmp_dir": self.config.temporary_dir,    # Temporary files location
            "test": kwargs.get("test", False),       # Test mode flag
            "task": str(task),                       # Task identifier
        }
        
        # Delegate evaluation execution to TaskManager
        print(f"Evaluation {task}")
        TaskManager.eval(task, evaluation_config)    

class Collectra(BaseROCrate):
    """Main workflow class with improved object-oriented design.
    
    Collectra is the primary interface for creating, managing, and executing
    machine learning workflows. It uses composition with specialized helper
    classes to handle configuration, file management, and execution logic.
    
    The class provides methods for:
    - Creating and loading workflows
    - Adding and removing tasks
    - Training and running workflows
    - Managing workflow persistence
    """

    def __init__(
        self,
        name: str,
        version: str,
        output: Path,
        file_format: Optional[str] = None,
        crate: Optional[ROCrate] = None,
    ):
        """Initialize a new Collectra workflow instance.
        
        Args:
            name: Unique name for the workflow
            version: Version string for workflow versioning
            output: Directory path for workflow output files
            file_format: Optional file format specification (defaults to 'grapto')
            crate: Optional existing ROCrate instance (creates new if None)
        """
        # Initialize ROCrate if not provided
        if crate is None:
            crate = ROCrate()

        # Initialize parent BaseROCrate class
        super().__init__(name, version, output, crate)

        # Create and validate configuration
        self._config = WorkflowConfig(name, version, output, file_format)
        self._config.validate()

        # Initialize helper classes using composition pattern
        self._file_manager = WorkflowFileManager(self._config)  # File operations
        self._executor = WorkflowExecutor(self._config)         # Execution logic

        # Provide user feedback about initialization
        print(f"Collectra workflow initialized: {self.name} version {self.version}")

        # Update ROCrate metadata with file format if specified
        if file_format:
            self.crate.update_jsonld(
                {
                    "@id": "./",
                    "file_format": file_format,
                }
            )

    @property
    def config(self) -> WorkflowConfig:
        """Get workflow configuration.
        
        Provides read-only access to the workflow configuration object.
        
        Returns:
            WorkflowConfig: The current workflow configuration
        """
        return self._config

    def run(self, input_path: Path, output_path: Path) -> None:
        """Execute the workflow for inference.
        
        Runs the complete workflow on the provided input data and saves
        results to the specified output location.
        
        Args:
            input_path: Path to input data for processing
            output_path: Path where results will be saved
        """
        # Delegate execution to the specialized executor
        self._executor.run_workflow(self.crate, input_path, output_path)

    def train(
        self, task: str, input_files: List[str], output_path: Path, **kwargs
    ) -> None:
        """Train the workflow with specified data.
        
        Executes the training process for a specific task, then automatically
        saves the updated workflow and cleans up temporary files.
        
        Args:
            task: String identifier for the task to train
            input_files: List of file paths containing training data
            output_path: Path where training results will be saved
            **kwargs: Additional training parameters (e.g., test=True)
        """
        # Execute training through the specialized executor
        self._executor.train_workflow(
            self.crate, task, input_files, output_path, **kwargs
        )
        
        # Automatically save the updated workflow after training
        self.save(compressed=True)
        
        # Clean up any temporary files created during training
        self.cleanup()

    def eval(
        self, task: str, input_files: List[str], output_path: Path, **kwargs
    ) -> None:
        """Evaluate the workflow with specified data.
        
        Executes the evaluation process for a specific task, then automatically
        saves the updated workflow and cleans up temporary files.
        
        Args:
            task: String identifier for the task to evaluate
            input_files: List of file paths containing evaluation data
            output_path: Path where evaluation results will be saved
            **kwargs: Additional evaluation parameters (e.g., test=True)
        """
        # Execute evaluation through the specialized executor
        self._executor.eval_workflow(
            self.crate, task, input_files, output_path, **kwargs
        )
        
        # Automatically save the updated workflow after evaluation
        self.save(compressed=True)
        
        # Clean up any temporary files created during evaluation
        self.cleanup()  

    def add_task(self, task: str) -> Optional[TaskEntity]:
        """Add a single task to the Collectra workflow.
        
        Parses the task definition string and creates a new TaskEntity
        that is added to the workflow's ROCrate.
        
        Args:
            task: The task definition in the format "<task_type>,<task_name>,<engine_type>,<engine>"
            
        Returns:
            TaskEntity: The created task entity, or None if creation failed
            
        Example:
            workflow.add_task("detect_object,my_detector,yolo,yolo11n.pt")
        """
        try:
            # Use TaskManager to build and validate the task
            return TaskManager.build(task, self.crate)
        except ValueError as e:
            # Handle validation errors gracefully
            print(f"[red]Error adding task[/red]: {e}")
            return None
    
    def edit_task(self, task: str, param: str, value: str) -> Optional[TaskEntity]:
        """Edit a specific task in the Collectra workflow.
        
        Modifies the specified parameter of the given task with the new value.
        This allows dynamic updates to task configurations without recreating them.
        
        Args:
            task: The task identifier to edit
            param: The parameter to modify
            value: The new value for the parameter
            
        """
        try:
            # Use TaskManager to build and validate the task
            return TaskManager.edit(task, self.crate, param, value)
        except ValueError as e:
            # Handle validation errors gracefully
            print(f"[red]Error editing task[/red]: {e}")
            return None
        

    def add_tasks(self, tasks: List[str]) -> List[TaskEntity]:
        """Add multiple tasks to the Collectra workflow.
        
        Convenience method for adding several tasks at once. Each task
        is processed individually using add_task().
        
        Args:
            tasks: List of task definition strings
            
        Returns:
            List[TaskEntity]: List of created task entities (may contain None for failed tasks)
            
        Example:
            workflow.add_tasks([
                "detect_object,detector1,yolo,yolo11n.pt",
                "classify_image,classifier1,image_classifier,resnet.pt"
            ])
        """
        task_crates = []
        # Process each task definition individually
        for task in tasks:
            # Add each task and collect results (including None for failures)
            task_crates.append(self.add_task(task))
        return task_crates

    def save(self, compressed: bool = True) -> None:
        """Save the current state of the Collectra workflow.
        
        Persists the workflow to disk using the configured output directory.
        The workflow can be saved as either a compressed ZIP file or an
        uncompressed directory structure.
        
        Args:
            compressed: If True, save as ZIP file; if False, save as directory
        """
        # Delegate file saving to the specialized file manager
        self._file_manager.save_workflow(self.crate, compressed)

    def delete_task(self, task_id: str) -> bool:
        """Remove a task from the Collectra workflow.
        
        Locates and removes the specified task from the workflow's ROCrate.
        Provides user feedback about the operation status.
        
        Args:
            task_id: The unique identifier of the task to remove
            
        Returns:
            bool: True if task was successfully removed, False otherwise
        """
        try:
            # Inform user of removal attempt
            print(f"[purple]Attempting to remove task[/purple]: {task_id}")
            
            # Locate the task in the ROCrate
            task = self.crate.dereference(task_id)
            
            # Remove the task from the ROCrate
            self.crate.delete(task)
            
            # Confirm successful removal
            print(f"[green]Task removed successfully.[/green]")
            return True
            
        except ValueError as e:
            # Handle errors gracefully (e.g., task not found)
            print(f"[red]Error removing task[/red]: {e}")
            return False

    def cleanup(self, tasks: Optional[List[TaskEntity]] = None) -> None:
        """Cleanup temporary files created during workflow execution.
        
        Removes temporary files and directories that were created during
        workflow operations. This helps maintain a clean workspace.
        
        Args:
            tasks: Optional list of specific TaskEntity instances to clean up.
                  If None, performs general cleanup of temporary directories.
        """
        # Delegate cleanup to the specialized file manager
        self._file_manager.cleanup_temporary_files(tasks or [])

    @staticmethod
    def make(
        name: str = "default",
        version: str = "1.0",
        output: Path = Path.cwd(),
        tasks: Optional[List[str]] = None,
        file_format: str = "grapto",
    ) -> Optional[Collectra]:
        """Create a new Collectra workflow with specified parameters.
        
        Factory method for creating and initializing a new workflow.
        Handles directory creation, task initialization, and file saving.
        
        Args:
            name: Name for the new workflow (defaults to 'default')
            version: Version string for the workflow (defaults to '1.0')
            output: Output directory path (defaults to current directory)
            tasks: Optional list of task definitions to add initially
            file_format: File format for the workflow (defaults to 'grapto')
            
        Returns:
            Collectra: New workflow instance, or None if creation failed
            
        Note:
            If a workflow with the same name exists (except 'default'),
            creation will be skipped to prevent accidental overwrites.
        """
        # Ensure tasks list is initialized
        if tasks is None:
            tasks = []

        # Inform user about workflow creation
        print(f"Creating workflow: {name} in {output}")
        workflow_path = Path(output) / name

        # Check for existing workflows (except default which can be overwritten)
        if workflow_path.exists() and name != "default":
            print(
                f"[red]Workflow file already exists[/red]: {name} - skipping initialisation. To edit use `collectra edit` command."
            )
            return None

        # Special handling for 'default' workflow - always recreate
        if name == "default":
            rmtree(workflow_path, ignore_errors=True)

        # Ensure output directory exists
        os.makedirs(output, exist_ok=True)

        try:
            # Create new workflow instance
            collectra = Collectra(name, version, output, file_format=file_format)
            # collectra._save_dependencies()  # Save dependencies
            
            # Add initial tasks if provided
            created_tasks = collectra.add_tasks(tasks)
            
            # Save the workflow to disk
            collectra.save()
            
            # Clean up any temporary files from task creation
            collectra.cleanup(created_tasks)
            
            return collectra
            
        except Exception as e:
            # Handle any errors during workflow creation
            print(f"[red]Error creating workflow[/red]: {e}")
            return None

    @staticmethod
    def load_workflow(workflow_file: Path) -> Optional[Collectra]:
        """Load an existing workflow from a file.
        
        Factory method for loading previously saved workflows from disk.
        Handles file validation and error recovery.
        
        Args:
            workflow_file: Path to the saved workflow file (ZIP or directory)
            
        Returns:
            Collectra: Loaded workflow instance, or None if loading failed
            
        Note:
            The loaded workflow will use the filename (without extension)
            as the workflow name and the parent directory as output location.
        """
        # Validate that the workflow file exists
        if not workflow_file.exists():
            print(
                f"[red]Error: The specified workflow does not exist[/red]: {workflow_file}"
            )
            return None

        try:
            # Load the ROCrate from the file
            crate = ROCrate(workflow_file)
            
            # Create Collectra instance with loaded crate
            collectra = Collectra(
                name=workflow_file.stem,        # Use filename as workflow name
                version="1.0",                  # Default version for loaded workflows
                output=workflow_file.parent,    # Use parent directory as output
                crate=crate,                    # Use the loaded crate
            )            
            return collectra
            
        except Exception as e:
            # Handle any errors during workflow loading
            print(f"[red]Error loading workflow[/red]: {e}")
            return None
