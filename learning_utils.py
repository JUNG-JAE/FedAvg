# ----------- System library ----------- #
import sys
from collections import defaultdict

# ----------- Learning library ----------- #
import torch
import torch.nn as nn

# ----------- Custom library ----------- #
from utils import create_directory, print_log
from conf import settings
from data_lodaer import source_dataloader


def get_network(args):
    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg11 import VGG
        net = VGG()
        # from models.vgg import vgg11_bn
        # net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'custom':
        from models.customCNN import CustomCNN
        net = CustomCNN()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net


def aggregation(args, models):
    aggregated_model = get_network(args)
    aggregated_model_dict = defaultdict(lambda: 0)

    coefficient = 1 / len(models)

    for model in models:
        for layer, params in model.state_dict().items():
            aggregated_model_dict[layer] += coefficient * params

    aggregated_model.load_state_dict(aggregated_model_dict)

    return aggregated_model


def save_model(args, global_round, model, model_name):
    save_path = f"{settings.LOG_DIR}/{settings.DATA_TYPE}/{args.net}/global_model/G{global_round}"
    create_directory(save_path)
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")


@torch.no_grad()
def source_evaluate(args, logger, model):
    _, test_loader = source_dataloader()

    loss_function = nn.CrossEntropyLoss()
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    device = torch.device('cuda' if args.gpu else 'cpu')
    model.to(device)

    test_loss = 0.0
    correct = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        _, predicted = torch.max(outputs, 1)
        c = (predicted == targets).squeeze()

        for i in range(len(targets)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        test_loss += loss.item()
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum()

    print_log(logger, 'Evaluating Model ... ')
    print_log(logger, f"Accuracy {correct.float() * 100 / len(test_loader.dataset):.2f}, Average loss: {test_loss / len(test_loader.dataset):.2f}")
    print_log(logger, '-------------------------------------')
    for i in range(10):
        print_log(logger, 'Accuracy of %3s : %2d %%' % (settings.LABELS[i], 100 * class_correct[i] / class_total[i]))
    print_log(logger, " ")

    return correct.float() * 100 / len(test_loader.dataset)

