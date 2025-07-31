import logging
import os
from datetime import datetime


def get_id():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def generate_run_path():
    res_dirname = "sketchmap-runs"
    os.makedirs(res_dirname, exist_ok=True)

    id = get_id()
    run_path = os.path.join(res_dirname, id)
    os.makedirs(run_path, exist_ok=True)

    print(f"Results will be saved to {run_path}")

    return run_path


def get_logger(save_path: str = "sketchmap-logs"):
    filename = os.path.join(save_path, f"logs.log")

    logging.basicConfig(
        filename=filename,
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.INFO,
    )

    return logging.getLogger()


RUN_PATH = generate_run_path()

logger = get_logger(RUN_PATH)
