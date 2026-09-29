import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Optional

from collectra import ObjectDetectionYOLO, Task
from collectra.utils import load_class_from_string
from utils.get_types import get_param_types, get_return_type


def test_generic_task(generic_type, debug):
    try:
        task = Task("generic_task")
        assert task.name == "generic_task", "Task name should be set correctly"
        param_types = get_param_types(task.run)
        assert param_types == dict(), "Parameter types should be empty for base Task"
        return_type = get_return_type(task.run)
        assert (
            return_type["return"] == Optional[generic_type]
        ), "Return type should be Generic for base Task"
    except Exception as e:
        debug(e)


def test_object_detection_task(task_data, image, classes, tmp_path, monkeypatch):
    """Exercise YOLO task construction and training orchestration cheaply."""

    class FakeYOLO:
        model_name = "fake.pt"

    fake_models = ModuleType("ultralytics.models")
    fake_models.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics.models", fake_models)

    name, data = task_data
    assert data[name], f"Task {name} should be in task_data"
    cls = load_class_from_string(data[name].pop("type"))
    task = cls(name=name, **data[name])
    assert isinstance(task, ObjectDetectionYOLO)

    crop = image.make_crop(0.5, 0.5, 0.25, 0.25, name="human")
    crop.add_source_parent(image)
    # Satisfy the backend type guard without loading weights or constructing a
    # neural network. External training itself is not a unit-test concern.
    task.model = FakeYOLO()
    monkeypatch.setattr(task, "_init_model", lambda: None)
    monkeypatch.setattr(task, "_prepare_assets", lambda *args: (["train"], ["val"]))
    seen = {}

    def fake_train_fold(train, validation, output_classes, log, kwargs):
        seen.update(
            train=train,
            validation=validation,
            classes=output_classes,
        )
        return SimpleNamespace(save_dir=log, results_dict={"mock": True})

    monkeypatch.setattr(task, "_train_fold", fake_train_fold)
    results = task._train(
        crop,
        classes=classes,
        project=f"{task.name}-test",
        log="run",
        base_folder=tmp_path,
    )

    assert results.results_dict == {"mock": True}
    assert seen == {
        "train": ["train"],
        "validation": ["val"],
        "classes": classes,
    }
