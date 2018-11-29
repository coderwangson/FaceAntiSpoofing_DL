import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from skimage import io
from load_imglist import ImageList
import random
import resnet as models
import os.path as osp
import numpy as np
import EER
from sklearn import metrics





model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names.append('mobilenet')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', #######################
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='model_best.pth.tar', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

#---------------------------------------------------------------------------
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 64)')



parser.add_argument('--root_path', type=str, default='/media/haoran/Data1/LivenessDetection/Data/CBSR-Antispoofing/',
                    metavar='H',
                    help='Dir Head')
parser.add_argument('--trainFile', type=str, default='label_img_train_all.txt', metavar='TRF', help='training file name')


#parser.add_argument('--trainFile', type=str, default='TestList_5_always.txt', metavar='TRF',help='training file name')
parser.add_argument('--testFile', type=str, default='label_img_test_all.txt', metavar='TEF',
                    help='test file name')
parser.add_argument('--da', type=int, default=5, metavar='DA',
                    help='data argumentations')

parser.add_argument('--depth', type=int, default=18, metavar='depth of resnet',
                    help='arch of resnet')


parser.add_argument('--logs_dir', type=str, default='./checkpoints/', metavar='path of checkpoint',
                    help='path of check point')


parser.add_argument('--gpu', type=int, default=[0,1], nargs='+', help='used gpu')





best_prec1 = 0
seed = int(random.random() * 1000)

def main():
    global args, best_prec1, seed
    args = parser.parse_args()


    #os.environ['CUDA_VISIBLE_DEVICES']=', '.join(str(x) for x in args.gpu)



    model=models.ResNet(depth=args.depth, pretrained=args.pretrained, cut_at_pooling=False, num_features=0, norm=False, dropout=0, num_classes=2)


    # # create model
    # if args.pretrained:  # from system models
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)   #from pytorch system





    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model=model.cuda()  ############################# important, it means model operation is conducted on cuda
    else:
        model=model()





    print ('Loading data from '+ args.root_path+args.trainFile)


    if args.da==0:
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5 ], std = [ 0.5, 0.5, 0.5 ]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.da==1:
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5 ], std = [ 0.5, 0.5, 0.5 ]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.da==2:
        deg=random.random()*10
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.ColorJitter(),
                                            transforms.RandomRotation(deg),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [ 0.5, 0.5, 0.5 ]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.da==3:
        deg=random.random()*10
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.RandomRotation(deg),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [ 0.5, 0.5, 0.5 ]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.da==4:
        deg=random.random()*20
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.ColorJitter(),
                                            transforms.RandomRotation(deg),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.da==5:
        deg=random.random()*10
        train_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.trainFile,
                      transform=transforms.Compose([
                                            transforms.RandomCrop(114),
                                            transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
                                            transforms.RandomRotation(deg),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [ 0.5, 0.5, 0.5 ]),
                                            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        pass

    print ('Loading data from ' + args.root_path + args.testFile)



    if args.da==5:
        val_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.testFile,
                      transform=transforms.Compose([
                                            transforms.CenterCrop(114),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5 ])
                                            ])),
            batch_size=args.test_batch_size, shuffle=False,  #### Shuffle should be switched off for face recognition of LFW
            num_workers=args.workers, pin_memory=True)

    else:
        val_loader = torch.utils.data.DataLoader(
            ImageList(root=args.root_path, fileList=args.root_path+args.testFile,
                      transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5 ])
                                            ])),
            batch_size=args.test_batch_size, shuffle=False,  #### Shuffle should be switched off for face recognition of LFW
            num_workers=args.workers, pin_memory=True)





    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True  # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.


    if args.evaluate:
        validate(val_loader, model, criterion)
        return



    fp = open(args.root_path + 'results_ResNet' + str(args.depth) + '_DA' + str(args.da) + '_LR_' + str(args.lr) + '.txt', "a")
    fp.write(str(seed)+'\n')
    fp.close()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)   # every epoch=10, the learning rate is divided by 10

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        fp = open(args.root_path + 'results_ResNet' + str(args.depth) + '_DA' + str(args.da) + '_LR_' + str(args.lr) + '.txt', "a")

        fp.write('{0:.3f} \n'.format(prec1))
        if epoch==args.epochs-1:
            fp.write('\n \n \n')
            fp.close()



        # remember best prec@1 and save checkpoint
        is_best = prec1 >best_prec1
        best_prec1 = max(prec1, best_prec1)
        #save_checkpoint({
        #    'epoch': epoch + 1,
        #    'arch': args.arch,
        #    'state_dict': model.state_dict(),
        #    'best_prec1': best_prec1,
        #    'optimizer' : optimizer.state_dict(),
        #}, is_best, filename=osp.join(args.logs_dir, 'checkpoint.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()  # variabls are set to 0
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    predict_sum = []
    target_sum = []
    prob = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target.cuda(async=True)

        input = input.cuda()
        target = target.cuda()

        #print input.get_device()


        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        outputs = model(input_var)
        loss = criterion(outputs, target_var)

        _,predict = torch.max(outputs.data, 1)
        predict_sum.extend(predict.cpu().numpy())
        target_sum.extend(target.cpu().numpy())
        prob.extend(outputs.data.cpu().numpy()[:,1])


        auc, eer, thd = EER.EER(target_sum, prob, 'CASIA')
        h = EER.HTER_NEW(target_sum, prob, 0.5, 'CASIA')
        auc, eer, thd = EER.EER_new(target_sum, prob, 'CASIA')
        all_acc_score = metrics.accuracy_score(target_sum, predict_sum)
        print ('Accuracy of the network on the test images is')
        print(all_acc_score)

        # # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        prec1, prec5 = accuracy(outputs.data, target, topk=(1, 1))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5=top1


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    predict_sum = []
    target_sum = []
    prob = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    #for i, sample_batched in enumerate(val_loader):
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        outputs = model(input_var)
        loss = criterion(outputs, target_var)

        _, predict = torch.max(outputs.data, 1)
        predict_sum.extend(predict.cpu().numpy())
        target_sum.extend(target.cpu().numpy())
        prob.extend(outputs.data.cpu().numpy()[:, 1])

        auc, eer, thd = EER.EER(target_sum, prob, 'CASIA')
        h = EER.HTER_NEW(target_sum, prob, 0.5, 'CASIA')
        auc, eer, thd = EER.EER_new(target_sum, prob, 'CASIA')
        all_acc_score = metrics.accuracy_score(target_sum, predict_sum)
        print('Accuracy of the network on the test images is')
        print(all_acc_score)


        prec1, prec5 = accuracy(outputs.data, target, topk=(1, 1))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5=top1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))




    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.logs_dir+'Best_'+str(seed)+'_ResNet'+str(args.depth)+'_LR'+str(args.lr)+'.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(outputs, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()  # view() for 2D tensor
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
