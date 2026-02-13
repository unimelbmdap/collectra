from abc import ABC, abstractmethod


class Converter(ABC):

    @abstractmethod
    def convert(self):
        raise NotImplementedError(
            "Converter subclasses must implement the convert method."
        )
