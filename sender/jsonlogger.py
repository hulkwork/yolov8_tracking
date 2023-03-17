from .base import BasSender
from loguru import logger
import json

class JsonLogger(BasSender):
    def __init__(self, log_filename:str = "tracking.log") -> None:
        super().__init__()
        self.logger = logger
        self.logger.add(log_filename, format="{message}", level="INFO")

    def send(self, messages):
        self.logger.info(json.dumps(messages))