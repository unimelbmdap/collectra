"""Tasks are told the results folder as an absolute path.

Tasks run inside the pipeline's folder, so a relative ``--output`` would point
somewhere else by then: files a task saves beside its results (such as masks)
would land in the pipeline's folder.
"""

import shutil
from pathlib import Path

from collectra import Pipeline
from collectra.tasks.base import Task
from collectra.types.images import Image
from collectra.types.texts import Text

THIS = "tests.unit.test_task_context_path"
SEEN: list[Path] = []


class Where(Task):
    def run(self, image: Image) -> Text:
        SEEN.append(self.context.file_path)
        return Text(name=self.get_output_name(), data="x")


def test_a_relative_output_folder_reaches_tasks_as_an_absolute_path(
    tmp_path, raw_img_path, monkeypatch
):
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(raw_img_path, work / "page.jpg")
    monkeypatch.chdir(work)
    SEEN.clear()
    Pipeline(
        "Where",
        "whr",
        "0.1.0",
        path=str(pipeline_dir),
        where={"type": f"{THIS}.Where", "input": ["page"], "output": ["note"]},
    ).run(files=["page.jpg"], output="out")
    assert SEEN == [work.resolve() / "out" / "page.whr"]
    assert SEEN[0].is_absolute()
