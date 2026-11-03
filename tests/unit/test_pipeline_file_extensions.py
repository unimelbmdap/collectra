from collectra import Artefact, Image, Pipeline, Text
from collectra.tasks.base import Task

THIS = "tests.unit.test_pipeline_file_extensions"


class FromImage(Task):
    def run(self, image: Image) -> Text:
        return Text(name=self.get_output_name(), data="x")


class FromText(Task):
    def run(self, text: Text) -> Text:
        return Text(name=self.get_output_name(), data=text())


class FromBoth(Task):
    def run(self, image: Image, text: Text) -> Text:
        return Text(name=self.get_output_name(), data=text())


class FromAny(Task):
    def run(self, thing: Artefact) -> Text:
        return Text(name=self.get_output_name(), data="x")


def pipeline(tmp_path, **tasks):
    return Pipeline("Demo", "dmo", "0.1.0", path=str(tmp_path), **tasks)


def test_a_pipeline_reads_its_own_extension_and_its_input_types(tmp_path):
    images = pipeline(
        tmp_path,
        read={"type": f"{THIS}.FromImage", "input": ["page"], "output": ["out"]},
    )
    assert images.file_extensions() == {".dmo"} | Image.file_extensions()


def test_a_text_pipeline_reads_text_files(tmp_path):
    texts = pipeline(
        tmp_path,
        read={"type": f"{THIS}.FromText", "input": ["note"], "output": ["out"]},
    )
    assert texts.file_extensions() == {".dmo"} | Text.file_extensions()


def test_several_input_nodes_leave_only_the_pipeline_extension(tmp_path):
    both = pipeline(
        tmp_path,
        read={
            "type": f"{THIS}.FromBoth",
            "input": ["page", "note"],
            "output": ["out"],
        },
    )
    assert both.file_extensions() == {".dmo"}


def test_a_pipeline_without_tasks_reads_only_its_own_extension(tmp_path):
    other = Pipeline("Demo", "dmo", "0.1.0", path=str(tmp_path))
    assert other.file_extensions() == {".dmo"}


def test_an_input_type_that_names_no_extensions_adds_none(tmp_path):
    anything = pipeline(
        tmp_path,
        read={"type": f"{THIS}.FromAny", "input": ["thing"], "output": ["out"]},
    )
    assert anything.file_extensions() == {".dmo"}
