import re
from typer.testing import CliRunner
from collectra import Collectra

from .test_main import (
    TEST_FILES_STR,
    DUMMY_WORKFLOW,
)

runner = CliRunner()

def run_app(command:str):
    workflow = Collectra.load_workflow(DUMMY_WORKFLOW)
    app = workflow.app()
    return runner.invoke(app, command.split())

def test_version():
    def assert_version_ok(result):
        assert result.exit_code == 0
        assert re.match(r"^(\d+\.)?(\d+\.)?(\*|\d+)$", result.stdout)

    assert_version_ok(run_app("--version"))
    assert_version_ok(run_app("-v"))


def test_train(tmpdir):
    result = run_app(f"train --task obj-detection1 {TEST_FILES_STR}")
    assert result.exit_code == 0

    output_obj1 = tmpdir/'output_obj1'
    result = run_app(f"train -t obj-detection1 --output {output_obj1} {TEST_FILES_STR}")
    assert result.exit_code == 0
    assert output_obj1.exists()
    output_obj1_log = output_obj1/'yolo.log' # CHANGE THIS AS NEEDED
    assert output_obj1_log.exists()
    output_obj1_log_text =  output_obj1_log.read_text()
    assert 'obj-detection1' in output_obj1_log_text # CHANGE THIS AS NEEDED

    output_obj2 = tmpdir/'output_obj2'
    result = run_app(f"train -t obj-detection2 {TEST_FILES_STR} -o {output_obj2}")
    assert result.exit_code == 0
    assert output_obj2.exists()
    output_obj2_log = output_obj2/'yolo.log' # CHANGE THIS AS NEEDED
    assert output_obj2_log.exists()
    output_obj2_log_text =  output_obj2_log.read_text()
    assert 'obj-detection2' in output_obj2_log_text # CHANGE THIS AS NEEDED


def test_eval(tmpdir):
    result = run_app(f"eval --task obj-detection1 {TEST_FILES_STR}")
    assert result.exit_code == 0
    assert "Evaluation obj-detection1" in result.stdout.strip()

    output_obj1 = tmpdir/'output_obj1'
    result = run_app(f"eval --task obj-detection1 --output {output_obj1} {TEST_FILES_STR}")
    assert result.exit_code == 0
    assert output_obj1.exists()
    output_obj1_log = output_obj1/'eval.log' # CHANGE THIS AS NEEDED
    assert output_obj1_log.exists()
    output_obj1_log_text =  output_obj1_log.read_text()
    assert 'obj-detection1' in output_obj1_log_text # CHANGE THIS AS NEEDED
    

def test_cluster(tmpdir):
    for index in range(1,3):
        output_cluster = tmpdir/'output_cluster'
        result = run_app(f"cluster --item label{index} {TEST_FILES_STR} --output {output_cluster}")
        assert result.exit_code == 0
        assert output_cluster.exists()
        output_cluster_log = output_cluster/'cluster.html' # CHANGE THIS AS NEEDED
        assert output_cluster_log.exists()
        output_cluster_log_text =  output_cluster_log.read_text()
        assert f'label{index}' in output_cluster_log_text # CHANGE THIS AS NEEDED
        

def test_extract(tmpdir):
    for index in range(1,3):
        output_extract = tmpdir/'output_extract'
        result = run_app(f"extract --item label{index} {TEST_FILES_STR} --output {output_extract}")
        assert result.exit_code == 0
        assert output_extract.exists()
        output_extract_log = output_extract/'extract.html' # CHANGE THIS AS NEEDED
        assert output_extract_log.exists()
        output_extract_log_text =  output_extract_log.read_text()
        assert f'label{index}' in output_extract_log_text # CHANGE THIS AS NEEDED
                