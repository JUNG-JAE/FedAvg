# ------------ System library ------------ #
import argparse

# ------------ custom library ------------ #
from worker import Worker
from utils import set_logger, print_log
from learning_utils import save_model, aggregation, source_evaluate
from conf import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    logger = set_logger(args)
    workers = [Worker(args, logger, f"worker{worker_index}") for worker_index in range(settings.NUM_OF_WORKER)]

    print(f"[ Number of worker: {len(workers)} ]")

    for global_round in range(settings.TOTAL_ROUND):
        print_log(logger, f"[ ========== Global Round: {global_round:2} ========== ]")
        model_list = []

        for worker in workers:
            print_log(logger, f"--------- Working ({worker.worker_id}) ---------")
            worker.load_global_model(global_round) if global_round != 0 else None
            worker.train()
            worker.evaluate()
            model_list.append(worker.model)
            save_model(args, global_round, worker.model, worker.worker_id)

            print_log(logger, " ")

        print_log(logger, "[Global Model accuracy]")
        aggregation_model = aggregation(args, model_list)
        source_evaluate(args, logger, aggregation_model)
        save_model(args, global_round, aggregation_model, "global_model")


if __name__ == '__main__':
    main()
