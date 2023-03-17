
class BasSender:
    def __init__(self) -> None:
        pass

    def send(self, messages):
        raise NotImplementedError()