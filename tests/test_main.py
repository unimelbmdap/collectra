import re, os, pytest, shutil
from pathlib import Path
from typer.testing import CliRunner
from collectra.tasks import ObjectDetectionYOLO
from collectra.models import YOLOModel, ImageClassifier
from collectra.main import app
from collectra.pipeline import CollectraManager

TEST_DATA = Path(__file__).parent / "testworkflow"
TEST_FILES = [str(file) for file in TEST_DATA.glob("*.hespi")]
TEST_FILES_STR = " ".join(TEST_FILES)
DUMMY_TASK_INPUT = {
    "task_name": "text_detect",
    "task_type": "object_detection",
    "engine_name": "yolo11n.pt",
    "engine_type": "yolo",
}
DUMMY_TASK_INPUT2 = {
    "task_name": "label_detect",
    "task_type": "object_detection",
    "engine_name": "yolo11n.pt",
    "engine_type": "yolo",
}

runner = CliRunner()


def run_app(command: str):
    return runner.invoke(app, command.split())


@pytest.fixture
def dummy_workflow(tmpdir):
    return Path(tmpdir) / "dummy_workflow"


def make_workflow(dummy_workflow):
    result = run_app(f"make -f hespi -w {dummy_workflow}")
    assert result.exit_code == 0
    assert dummy_workflow.exists(), "Workflow file should be created"
    task_string = ",".join(DUMMY_TASK_INPUT.values())
    result = run_app(f"add -w {dummy_workflow} -t {task_string}")
    assert result.exit_code == 0, f"Add task {task_string} should succeed"
    myworkflow = CollectraManager.load_workflow(dummy_workflow)
    return myworkflow


# def test_detect_object_task_and_engine_add():
#     """
#     Test the training functionality of the DetectObject task with a YOLOEngine.
#     """
#     engine = YOLOModel(name="yolo11n.pt", config={"test": 1, "file_format": "hespi", "input": TEST_FILES})
#     task = ObjectDetectionYOLO(task_type="detect_object", engine=engine)
#     assert isinstance(task.engine, YOLOModel), "Engine should be an instance of YOLOModel"
#     # Run the training process
#     task.train()

#     # Verify that the training files are created
#     yolo_config_file = Path(engine.dir) / "yolo_config.yml"
#     train_file = Path(engine.dir) / "train.txt"
#     val_file = Path(engine.dir) / "val.txt"

#     assert yolo_config_file.exists(), "YOLO config file should be created"
#     assert train_file.exists(), "Train file should be created"
#     assert val_file.exists(), "Validation file should be created"

# # def test_version():
# #     def assert_version_ok(result):
# #         assert result.exit_code == 0
# #         assert re.match(r"^(\d+\.)?(\d+\.)?(\*|\d+)$", result.stdout)

# #     assert_version_ok(run_app("--version"))
# #     assert_version_ok(run_app("-v"))

# def test_make(tmpdir, dummy_workflow):
#     """
#     Test the make command to ensure it creates the necessary directories and files.
#     """
#     result = run_app(f"make -f hespi -o {tmpdir}")
#     assert result.exit_code == 0
#     workflow_path = Path(tmpdir) / "default"
#     assert workflow_path.exists(), "Workflow file should be created"
#     assert workflow_path.is_file(), "Workflow path should be a file"
#     result = run_app(f"make -f hespi -w {dummy_workflow} -o {tmpdir}")
#     assert result.exit_code == 0
#     assert dummy_workflow.exists(), "Workflow file should be created in specified output directory"
#     assert dummy_workflow.is_file(), "Workflow path should be a file in specified output directory"


# def test_add_task(dummy_workflow):
#     """
#     Test the add command to ensure it adds tasks to the workflow.
#     """
#     myworkflow = make_workflow(dummy_workflow)
#     task = myworkflow.create.dereference(DUMMY_TASK_INPUT["task_name"])
#     assert task is not None, "Task should exists in the workflow"
#     assert task.get("task_type") == DUMMY_TASK_INPUT["task_type"], "Task type should be 'detect_object'"
#     assert task.get("engine")[0].id == DUMMY_TASK_INPUT["engine_name"], "Engine name should be 'yolo11n.pt'"
#     assert task.get("engine")[0].get("engine_type") == DUMMY_TASK_INPUT["engine_type"], "Engine type should be 'yolo'"

# def test_train(tmpdir, dummy_workflow):
#     make_workflow(dummy_workflow)
#     task_string2 = ",".join(DUMMY_TASK_INPUT2.values())
#     result = run_app(f"add -w {dummy_workflow} -t {task_string2}")
#     assert result.exit_code == 0, f"Add task {task_string2} should succeed"
#     result = run_app(f"train --workflow {dummy_workflow} --task {DUMMY_TASK_INPUT['task_name']} {TEST_FILES_STR} -te 1")
#     assert result.exit_code == 0

#     output_obj1 = Path(tmpdir) / 'output_obj1'
#     result = run_app(f"train -w {dummy_workflow} -t {DUMMY_TASK_INPUT['task_name']} --output {output_obj1} {TEST_FILES_STR} -te 1")
#     assert result.exit_code == 0
#     assert output_obj1.exists()
#     output_obj1_log = output_obj1 / 'yolo.log'  # CHANGE THIS AS NEEDED
#     assert output_obj1_log.exists()
#     output_obj1_log_text =  output_obj1_log.read_text()
#     assert f"{DUMMY_TASK_INPUT['task_name']}" in output_obj1_log_text # CHANGE THIS AS NEEDED

#     output_obj2 = Path(tmpdir) /'output_obj2'
#     result = run_app(f"train -w {dummy_workflow} -t {DUMMY_TASK_INPUT2['task_name']} {TEST_FILES_STR} -o {output_obj2} -te 1")
#     assert result.exit_code == 0
#     assert output_obj2.exists()
#     output_obj2_log = output_obj2 / 'yolo.log' # CHANGE THIS AS NEEDED
#     assert output_obj2_log.exists()
#     output_obj2_log_text =  output_obj2_log.read_text()
#     assert f"{DUMMY_TASK_INPUT2['task_name']}" in output_obj2_log_text # CHANGE THIS AS NEEDED


# def test_eval(tmpdir, dummy_workflow):
#     make_workflow(dummy_workflow)
#     result = run_app(f"eval --workflow {dummy_workflow} --task {DUMMY_TASK_INPUT['task_name']} {TEST_FILES_STR} -te 1")
#     assert result.exit_code == 0
#     assert f"Evaluation {DUMMY_TASK_INPUT['task_name']}" in result.stdout.strip()
#     output_obj1 = Path(tmpdir) /'output_obj1'
#     result = run_app(f"eval --workflow {dummy_workflow} --task {DUMMY_TASK_INPUT['task_name']} --output {output_obj1} {TEST_FILES_STR} -te 1")
#     assert result.exit_code == 0
#     assert output_obj1.exists()
#     output_obj1_log = output_obj1 / 'eval.log' # CHANGE THIS AS NEEDED
#     assert output_obj1_log.exists()
#     output_obj1_log_text =  output_obj1_log.read_text()
#     assert f"{DUMMY_TASK_INPUT['task_name']}" in output_obj1_log_text # CHANGE THIS AS NEEDED

# def test_edit(dummy_workflow):
#     make_workflow(dummy_workflow)
#     result = run_app(f"edit --workflow {dummy_workflow} --task {DUMMY_TASK_INPUT['task_name']} --param epochs --value 1")
#     assert result.exit_code == 0
#     result = run_app(f"view -w {dummy_workflow}")
#     assert result.exit_code == 0
#     assert "'epochs': 1" in result.stdout.strip(), "Task parameter 'epochs' should be updated to 1"

# def test_cluster(tmpdir, dummy_workflow):
#     make_workflow(dummy_workflow)
#     result = run_app(f"edit --workflow {dummy_workflow} --task {DUMMY_TASK_INPUT['task_name']} --param epochs --value 1")
#     assert result.exit_code == 0
#     output_obj1 = Path(tmpdir) / 'output_obj1'
#     result = run_app(f"train -w {dummy_workflow} -t {DUMMY_TASK_INPUT['task_name']} --output {output_obj1} {TEST_FILES_STR}")
#     assert result.exit_code == 0

#     labels = ["label1", "label2", "label3"]

#     for label in labels:
#         output_cluster = Path(tmpdir) / 'output_cluster'
#         result = run_app(f"cluster --workflow {dummy_workflow} --item {label} {TEST_FILES} --output {output_cluster}")
#         assert result.exit_code == 0
#         # assert output_cluster.exists()
#         # output_cluster_log = output_cluster/'cluster.html' # CHANGE THIS AS NEEDED
#         # assert output_cluster_log.exists()
#         # output_cluster_log_text =  output_cluster_log.read_text()
#         # assert f'label{index}' in output_cluster_log_text # CHANGE THIS AS NEEDED


# # def test_extract(tmpdir):
# #     for index in range(1,3):
# #         output_extract = tmpdir/'output_extract'
# #         result = run_app(f"extract --workflow {DUMMY_WORKFLOW} --item label{index} {TEST_FILES} --output {output_extract}")
# #         assert result.exit_code == 0
# #         assert output_extract.exists()
# #         output_extract_log = output_extract/'extract.html' # CHANGE THIS AS NEEDED
# #         assert output_extract_log.exists()
# #         output_extract_log_text =  output_extract_log.read_text()
# #         assert f'label{index}' in output_extract_log_text # CHANGE THIS AS NEEDED
