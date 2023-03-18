from .base import BasSender, NumpyArrayEncoder
import zmq
import json


class ZmqLogger(BasSender):
    def __init__(self, ip_addr:str = "localhost", port:int = 5555) -> None:
        super().__init__()
        self.context = zmq.Context()
        self.producer = self.context.socket(zmq.PUB)
        self.producer.connect(f"tcp://{ip_addr}:{port}")
        
    def send(self, messages):
        self.producer.send_string(json.dumps(messages, cls = NumpyArrayEncoder))