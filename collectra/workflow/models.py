class Builder:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def build(self):
        print(f"Building {self.name} version {self.version}")