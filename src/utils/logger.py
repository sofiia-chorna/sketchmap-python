import datetime
import logging


def get_logger():
    id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logging.basicConfig(
        filename=f"logs_{id}.log", format="%(asctime)s %(message)s", filemode="w"
    )

    return logging.getLogger()


logger = get_logger()
