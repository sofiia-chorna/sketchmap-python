import logging
import os
from datetime import datetime


def get_logger():
    dirpath = "sketchmap-python-logs"
    os.makedirs(dirpath, exist_ok=True)
    id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        filename=os.path.join(dirpath, f"logs_{id}.log"),
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    return logging.getLogger()


logger = get_logger()
