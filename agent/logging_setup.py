# agent/logging_setup.py
import logging.config
import yaml

def setup_logging():
    with open("config/logger.yaml", "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)

