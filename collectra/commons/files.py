from pathlib import Path

import pydantic

from ..utils import write_yaml


class CollectraResultsMetadata(pydantic.BaseModel):
    workflow: str | None = None
    version: str | None = None
    partition: str | None = None
    timestamp: str | None = None

    def model_dump(self, *args, **kwargs) -> dict:
        from datetime import datetime

        data = super().model_dump(*args, **kwargs)
        data["timestamp"] = datetime.now().isoformat()
        if self.partition is None:
            data.pop("partition")
        if self.workflow is None:
            data.pop("workflow")
        if self.version is None:
            data.pop("version")
        return data


class CollectraFile(pydantic.BaseModel):
    collectra_results_metadata: CollectraResultsMetadata
    data: dict

    def model_dump(self, *args, **kwargs) -> dict:
        data = super().model_dump(*args, **kwargs)
        for key, value in data["data"].items():
            data[key] = value
        data.pop("data")
        return data

    @classmethod
    def from_data(cls, file: Path):
        import yaml

        results_yaml = file / "results.yaml"
        if not results_yaml.exists():
            raise FileNotFoundError(f"results.yaml not found in {file}")
        with open(results_yaml, "r") as f:
            data = yaml.safe_load(f)
        metadata = CollectraResultsMetadata(**data.pop("collectra_results_metadata"))
        return cls(collectra_results_metadata=metadata, data=data)

    @classmethod
    def from_file(cls, file: Path, label: str, ext: str, force: bool = False):
        parent_dir = file.parent
        collectra_file_path = parent_dir / f"{file.stem}.{ext.strip('.')}"
        if collectra_file_path.exists() and not force:
            raise FileExistsError(
                f"Directory {collectra_file_path} already exists. To overwrite, use -f or --force option."
            )
        else:
            collectra_file_path.mkdir(parents=True, exist_ok=True)
        metadata = CollectraResultsMetadata()
        new_data = {f"{label}": {"type": "collectra.Image", "data": str(file.name)}}
        return cls(collectra_results_metadata=metadata, data=new_data)

    def save(self, file: Path):
        data = self.model_dump()
        write_yaml(data, file / "results.yaml")
