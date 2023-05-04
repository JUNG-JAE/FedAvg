# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torch.optim as optim

# ------------ system library ------------ #
from tqdm import tqdm

# ------------ custom library ------------ #
from conf import settings
from utils import print_log
from learning_utils import get_network
from data_lodaer import worker_dataloader, source_dataloader


class Worker:
    def __init__(self, args, logger, worker_id):
        self.worker_id = worker_id
        self.args = args
        self.logger = logger
        self.model = get_network(args)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        self.train_loader, self.test_loader = worker_dataloader(self.worker_id)
        # self.train_loader, self.test_loader = source_dataloader()
        self.device = torch.device('cuda' if args.gpu else 'cpu')

    def load_global_model(self, global_round):
        model_path = f"{settings.LOG_DIR}/{settings.DATA_TYPE}/{self.args.net}/global_model/G{global_round-1}/global_model.pt"
        self.model.load_state_dict(torch.load(model_path))

    def train(self, save_model=True):
        print_log(self.logger, "Training Model ... ")
        self.model.train()

        for epoch in range(settings.LEARNING_EPOCH):
            progress = tqdm(total=len(self.train_loader.dataset), ncols=100)

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()

                progress.update(settings.BATCH_SIZE)

            progress.close()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        test_loss = 0.0
        correct = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)

            test_loss += loss.item()
            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        print_log(self.logger, 'Accuracy: {:.4f}, Average loss: {:.4f}'.format(correct.float() * 100 / len(self.test_loader.dataset), test_loss / len(self.test_loader.dataset)))

        return correct.float() * 100 / len(self.test_loader.dataset)
