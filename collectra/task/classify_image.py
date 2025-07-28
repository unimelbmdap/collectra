from .base import Task
from .engine import Engine

class ImageClassfier(Engine):
    pass

class ClassifyImage(Task):

    VALID_ENGINES = (
        ImageClassfier
    )

    def __init__(self, engine: Engine):
        super().__init__()
        self.validate_engine(engine)        

    def validate_engine(self, engine: Engine):
        if not isinstance(engine, self.VALID_ENGINES):
            raise ValueError(f"Invalid engine type: {type(engine).__name__}. "
                             f"Valid engines are: {[e.__name__ for e in self.VALID_ENGINES]}.")
        self.engine = engine