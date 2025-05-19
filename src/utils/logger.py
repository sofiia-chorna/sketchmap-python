import logging
import os
from datetime import datetime


def get_id():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_logger():
    dirpath = "sketchmap-python-logs"
    os.makedirs(dirpath, exist_ok=True)

    id = get_id()
    filename = os.path.join(dirpath, f"logs_{id}.log")

    print(f"Logs are saved to {filename}")

    logging.basicConfig(
        filename=filename,
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    return logging.getLogger()


logger = get_logger()
