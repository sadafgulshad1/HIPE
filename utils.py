import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('a')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


import os
import torch
import pickle
import numpy as np
import pandas as pd

# ==================== General utils ====================#
def makedir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
def makepath(path):
    makedir(os.path.dirname(path))
    
def save_predictions(preds, pred_path):
    makepath(pred_path) 
    with open(pred_path,'wb') as f: pickle.dump(preds, f)
        
def save_checkpoint(state, checkpoint_path):
    makepath(checkpoint_path)
    torch.save(state, checkpoint_path)
    
def record_info(info, filename):
    print(''.join(['{}: {}   '.format(k,v) for k,v in info.items()]))
    
    df = pd.DataFrame.from_dict(info)
    column_names = list(info.keys())
    
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False, columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False, columns=column_names) 
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0] if len(topk) == 1 else res
               
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
        
# image utils
def tensor2img(tensor):
    return tensor.permute(1,2,0).cpu().detach().numpy()

def img2uint8(array):
    return (array * 255).astype(np.uint8)

def tensor2uint8(tensor):
    if tensor.shape[0] == 3:
        return img2uint8(tensor2img(tensor))
    else:
        img = img2uint8(tensor2img(tensor))[:,:,0]
        return np.stack([img,img,img], axis=2)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
def label_set_():
    
    csv_path = "UCF-101_hierarcy.csv"
    with open (csv_path,'r') as fp:
        lines = fp.readlines()
        labels = []
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            tmp_list = line.split(',')
            labels.append(tmp_list[0])
    label_set = list(set(labels))
    # label_set = list(set(class_names))
    label_set.sort() # 有Sort很重要
    
    return label_set