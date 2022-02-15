import click
import requests
import torch
import vgg

import pandas as pd

from os import walk
from torchvision import datasets, transforms
from models import MultiLayerNN
from tqdm import tqdm

import torch
import math
from torch.autograd import Variable
import numpy as np

import pickle

from sklearn.datasets import load_wine, load_boston
from sklearn.model_selection import train_test_split


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH,
                           params,
                           v,
                           learning_rate=1.,
                           id_subtract=False):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    if id_subtract:
        return group_add(v, hv, alpha=-learning_rate)
    else:
        return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self,
                 model,
                 learning_rate,
                 criterion,
                 data=None,
                 dataloader=None,
                 cuda=True,
                 id_subtract=False,
                 filename=""):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None
                and dataloader == None) or (data == None
                                            and dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.id_subtract = id_subtract
        self.filename = filename

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device)
               for p in self.params]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv

    def eigenvalues(self, maxIter=200, tol=1e-32, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device)
                 for p in self.params]  # generate random vector
            v = normalization(v)  # normalize the vector

            pbar = tqdm(range(maxIter))
            for i in pbar:
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(
                        self.gradsH,
                        self.params,
                        v,
                        learning_rate=self.learning_rate,
                        id_subtract=self.id_subtract)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    error = abs(eigenvalue -
                                tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                    pbar.set_description(
                        "Network: %s, power iter error: %.4f" %
                        (self.filename, error))
                    if error < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors


def has_dataset_name(f):
    if 'mnist' in f:
        return True
    elif 'cifar10' in f:
        return True
    elif 'cifar100' in f:
        return True
    elif 'svhn' in f:
        return True
    elif 'wine' in f:
        return True
    elif 'bhp' in f:
        return True
    else:
        return False


def load_data(dataset, batch_size):

    # mean/std stats
    if dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {'mean': [0.491, 0.482, 0.447], 'std': [0.247, 0.243, 0.262]}
    elif dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    elif dataset == 'svhn':
        data_class = 'SVHN'
        num_classes = 10
        input_dim = 32 * 32 * 3
        stats = {
            'mean': [-0.2159, -0.1573, 0.0985],
            'std': [0.4862, 0.5067, 0.4015]
        }
    elif dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        input_dim = 28 * 28
        stats = {'mean': [0.1307], 'std': [0.3081]}
    elif dataset == 'wine':
        data_class = 'WINE'
        num_classes = 3
        input_dim = 13
        stats = {'mean': [0], 'std': [1]}
    elif dataset == 'bhp':
        data_class = 'BHP'
        num_classes = 3
        input_dim = 13
        stats = {'mean': [0], 'std': [1]}
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(), lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
    ]

    if dataset not in ['wine', 'bhp']:
        # get tr and te data with the same normalization
        # no preprocessing for now
        try:
            tr_data = getattr(datasets,
                              data_class)(root='~/data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose(trans))
        except TypeError:
            tr_data = getattr(datasets,
                              data_class)(root='~/data',
                                          split='train',
                                          download=True,
                                          transform=transforms.Compose(trans))
    elif dataset == 'wine':
        X, y = load_wine(return_X_y=True)
        X, y = X.astype(np.float64), y.astype(np.float64)
        X = (X - X.mean(0)) / X.std(0)
        X_train, X_val, y_train, y_val = train_test_split(X,
                                                          y,
                                                          test_size=0.5,
                                                          stratify=y)
        tr_data = torch.utils.data.TensorDataset(
            torch.Tensor(X_train).type(torch.float32),
            torch.Tensor(y_train).type(torch.long))  # create your datset
    elif dataset == 'bhp':
        X, y = load_boston(return_X_y=True)
        X, y = X.astype(np.float64), y.astype(np.float64)
        X = (X - X.mean(0)) / X.std(0)
        idx = len(X) // 2
        X_train, _, y_train, _ = X[:idx], X[idx:], y[:idx], y[idx:]

        tr_data = torch.utils.data.TensorDataset(
            torch.Tensor(X_train).type(torch.float32),
            torch.Tensor(y_train).type(torch.float32).view(
                -1, 1))  # create your datset

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=batch_size,
        shuffle=False,
    )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, train_loader_eval, num_classes


@click.command()
# @click.argument('folder')
@click.option('--folder',
              '-f',
              required=True,
              help='Folder containing saved models',
              type=str)
@click.option('--batch_size',
              '-b',
              required=True,
              help='Batch size used to train model',
              type=int)
@click.option('--learning_rate',
              '-l',
              required=True,
              help='Learning rate used to train model',
              type=float)
@click.option('--dataset',
              '-d',
              help='Dataset used to train model',
              required=True,
              type=click.Choice(
                  ['cifar10', 'cifar100', 'mnist', 'svhn', 'wine', 'bhp']))
@click.option('--max_batch',
              '-m',
              help='Maximum number of batches to iterate over',
              required=False,
              default=None,
              type=int)
@click.option('--gpu',
              '-g',
              help='Use gpu',
              required=False,
              default=False,
              type=bool)
@click.option('--bound_est',
              '-e',
              help='Whether to calc eig(1-H) or 1 - max(eig(H))',
              required=False,
              default=False,
              type=bool)
@click.option('--save_folder',
              '-s',
              help='Folder to save to',
              required=False,
              default='.',
              type=str)
@click.option('--start_index',
              '-i',
              help='Start index of networks',
              required=False,
              default=0,
              type=int)
@click.option('--end_index',
              '-e',
              help='End index of networks',
              required=False,
              default=201,
              type=int)
def main(folder, batch_size, learning_rate, dataset, max_batch, gpu, bound_est,
         save_folder, start_index, end_index):
    """
    A tool to calculate the Jacobian of SGD updates.
    Provide a folder name, batch size, and learning rate.
    """
    if folder is None:
        folder = './'

    train_loader, train_loader_eval, num_classes = load_data(
        dataset, batch_size)

    _, _, filenames = next(walk(folder))
    filenames = [f for f in filenames if '.pyT' in f]
    filenames.sort()

    results = dict()

    for file in filenames[start_index:end_index]:
        print('>>>>>> finding dominant eigenvalues for ', file, ' >>>>>> ')
        dir = folder + '/' + file
        #right now I load them to cpu, but load them to gpu for hessian etc computation
        net = torch.load(dir, map_location='cpu')
        #our loss is always CE as we trained classifiers
        if dataset == 'bhp':
            loss = torch.nn.MSELoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        results[file] = []

        for i, data in enumerate(train_loader):
            print('-- batch %d --' % i)

            h = hessian(net,
                        learning_rate=learning_rate,
                        criterion=loss,
                        data=data,
                        cuda=gpu,
                        id_subtract=not bound_est,
                        filename=file)
            dom_eig, _ = h.eigenvalues(top_n=1)

            if bound_est:
                dom_eig[0] = max(1 - learning_rate * abs(dom_eig[0]),
                                 learning_rate * abs(dom_eig[0]) - 1)

            results[file].append(dom_eig[0])
            if (max_batch is not None) and (i + 1) == max_batch:
                break

    df = pd.DataFrame.from_dict(results,
                                orient='index',
                                columns=["batch %d" % j for j in range(i + 1)])

    #load the training/testing accuracy, loss etc
    run_name = [f for f in folder.split('/') if has_dataset_name(f)]

    with open(
            save_folder + '/' + run_name[-1] + '_iters_' + str(start_index) +
            'to' + str(end_index) + '_dom_eig.pkl', 'wb') as fp:
        res = dict(dom_eig=df)

        pickle.dump(res, fp)

    trn_res_list = torch.load(folder +
                              '/evaluation_history_extra_iters_TRAIN.hist',
                              map_location='cpu')
    tes_res_list = torch.load(folder +
                              '/evaluation_history_extra_iters_TEST.hist',
                              map_location='cpu')

    with open(
            save_folder + '/' + run_name[-1] + '_iters_' + str(start_index) +
            'to' + str(end_index) + '_dom_eig.pkl', 'wb') as fp:
        res = dict(dom_eig=df,
                   trn_res_list=trn_res_list,
                   tes_res_list=tes_res_list)

        pickle.dump(res, fp)


if __name__ == "__main__":
    main()
