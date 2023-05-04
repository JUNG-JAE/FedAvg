# ----------- System library ----------- #
import os
import pathlib
import logging

# ----------- Custom library ----------- #
from conf import settings


def set_global_round(args):
    global_round_path = pathlib.Path(settings.LOG_DIR) / settings.DATA_TYPE / args.net / "global_model"

    if not global_round_path.exists():
        print("[ ==================== Global Round: 1 ==================== ]")
        global_round = 1
    else:
        rounds = [int(p.name[1:]) for p in global_round_path.glob("G*")]
        global_round = max(rounds) + 1
        print(f"[ ==================== Global Round: {global_round:2} ==================== ]")

    new_round_path = global_round_path / f"G{global_round}"
    try:
        new_round_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        print(f"Error: Creating global model {global_round:2} directory")

    return global_round


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_logger(args):
    create_directory(f"{settings.LOG_DIR}/{settings.DATA_TYPE}/{args.net}/logs/")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=f"{settings.LOG_DIR}/{settings.DATA_TYPE}/{args.net}/logs/worker({settings.NUM_OF_WORKER})_epoch({settings.LEARNING_EPOCH})_round({settings.TOTAL_ROUND})_batch({settings.BATCH_SIZE})_rate({settings.LEARNING_RATE}).log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def print_log(logger, msg):
    print(msg)
    logger.info(msg)
