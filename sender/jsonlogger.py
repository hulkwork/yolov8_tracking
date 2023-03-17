from .base import BasSender
from loguru import logger
import json
from json import JSONEncoder
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

class JsonLogger(BasSender):
    def __init__(self, log_filename:str = "tracking.log") -> None:
        super().__init__()
        self.logger = logger
        self.logger.add(log_filename, format="{message}", level="INFO")

    def send(self, messages):
        self.logger.info(json.dumps(messages, cls=NumpyArrayEncoder))