import sys
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# from efficientnet_pytorch import EfficientNet

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


####
def prob2llr(p1x, prior1=0.5, eps=1e-100):
    # llr = log10( p1x/(1-p1x) ) + log10( (1-prior1) / prior1 )
    p1x = np.clip(p1x, eps, 1 - eps)
    prior1 = np.clip(prior1, eps, 1 - eps)
    llr = np.log10(p1x / (1 - p1x)) + np.log10((1 - prior1) / prior1)
    return llr


####
def test_classification(model, loader, dtype, num_classes):
    model.eval()
    prob_scores = dict()
    llr_scores = dict()
    c = 0

    # set batch=1 to retrieve sample_fname
    for i, (x, y) in enumerate(loader, 0):
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)[:, :num_classes].data.cpu().detach()

        probs = nn.functional.softmax(scores, dim=1)
        sample_fname, _ = loader.dataset.samples[i]
        _, preds = scores.data.cpu().max(1)
        scores = scores.numpy()
        probs = probs.numpy()
        preds = preds.numpy()

        prob_scores[sample_fname] = float(probs[0][0])
        llr_scores[sample_fname] = prob2llr(1.0 - probs[0][0])
        print('### ', c, sample_fname, prob_scores[sample_fname], llr_scores[sample_fname], preds[0], probs[0][:num_classes], scores[0][:num_classes])
        c += 1

    return llr_scores, prob_scores


####
def generated_image_classification(args):
    print('run testing on %s' % args.test_dir)

    dtype = torch.FloatTensor
    if args.use_gpu:
        dtype = torch.cuda.FloatTensor

    test_transform = T.Compose([
        T.Resize(args.tile),
        T.CenterCrop(args.tile),
        T.ToTensor(),
        T.Normalize(mean=args.mean, std=args.std)
    ])
    test_dset = ImageFolder(args.test_dir, transform=test_transform)
    num_classes = 2
    print('local test model name %s' % args.test_model_name)

    test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = nn.Module()
    pretrain_flag = False
    if args.model_arch.lower() == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrain_flag)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_arch.lower() == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrain_flag)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_arch.lower() == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrain_flag)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_arch.lower() == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrain_flag)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_arch.lower() == 'alexnet':
        model = torchvision.models.alexnet(pretrained=pretrain_flag)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif args.model_arch.lower() == 'vgg':
        model = torchvision.models.vgg16(pretrained=pretrain_flag)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif args.model_arch.lower() == 'squeezenet':
        model = torchvision.models.squeezenet1_0(pretrained=pretrain_flag)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif args.model_arch.lower() == 'densenet':
        model = torchvision.models.densenet121(pretrained=pretrain_flag)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif args.model_arch.lower() in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                                     'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']:
        model = EfficientNet.from_pretrained(args.model_arch.lower())
        model._fc.out_features = num_classes
    else:
        print('unknown model_arch %s' % args.model_arch)
        sys.exit(-1)

    model.type(dtype)

    checkpoint = torch.load(args.test_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    # print(model)
    llr_scores, prob_scores = test_classification(model, test_loader, dtype, num_classes)
    print(llr_scores)
    print(prob_scores)
    return llr_scores


####
if __name__ == '__main__':
    usage = "usage: python generated_image [args]\n"
    usage += "  Use pytorch to classify images and evaluate testing performance."
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--tile', default=224, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--model_arch', default='resnet101',
                        help="Network archtecture: efficientnet-b0, efficientnet-b5, resnet18, resnet34, resnet50, resnet101, vgg, squeezenet, densenet, alexnet")
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--test_model_name', default='cp00100.pt')
    parser.add_argument('--mininterval', default=60, type=float)
    parser.add_argument('--shift', default=0, type=int)  # use 128 for residual image
    parser.add_argument('--mean', type=lambda s: [float(item) for item in s.split(',')], default='0.502,0.502,0.502')
    parser.add_argument('--std', type=lambda s: [float(item) for item in s.split(',')], default='0.0446,0.0450,0.0442')

    args = parser.parse_args()
    print(args)

    if args.shift > 0:
        print('shift = ', args.shift)
        a = float(args.shift) / 255.0
        args.mean = [a, a, a]
        a = 1.0 / 255.0
        args.std = [a, a, a]

    print('mean = {}'.format(args.mean))
    print('std  = {}'.format(args.std))

    generated_image_classification(args)
