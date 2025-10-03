from pathlib import Path
from collectra.commons import MetaClass

class Type(MetaClass):
    def serialize(self) -> dict:
        serialized = dict(type=self.get_class_path())
        for key, value in self.attributes.items():
            if isinstance(value, Path):
                value = str(value)
            serialized[key] = value
        return serialized

    @property
    def attributes(self) -> dict:
        ignore = self.attributes_to_ignore()
        return {k:v for k, v in self.__dict__.items() if k not in ignore}
    
    def attributes_to_ignore(self) -> set:
        return set()

    @classmethod
    def get_class_path(cls: type) -> str:
        """Return the fully qualified path for a class.
        
        Example:
            >>> from collectra.images import Image
            >>> get_class_path(Image)
            'collectra.images.Image'
        """
        return f"{cls.__module__}.{cls.__qualname__}"
