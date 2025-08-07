from pathlib import Path
from collectra.parsing import read_collectra, render_collectra

TEST_DATA = Path(__file__).parent/"test-data"


def test_basic():
    basic_yaml_path = TEST_DATA / "basic.yaml"
    graph = read_collectra(basic_yaml_path)
    assert len(graph.nodes) == 5


def test_basic_render(tmpdir):
    basic_yaml_path = TEST_DATA / "basic.yaml"
    output_path = tmpdir / "basic.svg"
    dot_string = render_collectra(basic_yaml_path, output_path)
    assert "digraph" in dot_string  # Check if the output is a valid DOT string
    assert "processor2 -> output_text;" in dot_string

    assert output_path.exists()
    svg = output_path.read_text(encoding='utf-8')
    assert "<svg" in svg  # Check if the output file is a valid SVG
    assert "processor2" in svg  # Check if the SVG contains the expected node


def test_object_detection():
    object_detection_yaml_path = TEST_DATA / "object_detection.yaml"
    graph = read_collectra(object_detection_yaml_path)
    assert len(graph.nodes) == 9



def test_object_detection_render(tmpdir):
    object_detection_yaml_path = TEST_DATA / "object_detection.yaml"
    output_path = tmpdir / "object_detection.svg"

    dot_string = render_collectra(object_detection_yaml_path, output_path)

    assert "digraph" in dot_string  # Check if the output is a valid DOT string
    assert "input_image -> primary_label_detection;" in dot_string

    assert output_path.exists()
    svg = output_path.read_text(encoding='utf-8')
    assert "<svg" in svg  # Check if the output file is a valid SVG
    assert "primary_label" in svg  # Check if the SVG contains the expected node


