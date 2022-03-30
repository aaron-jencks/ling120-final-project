import torch
from tqdm import tqdm

from typing import List


class GeneralPerceptron(torch.nn.Module):
    def __init__(self, n_param: int, n_out: int = 1,
                 n_layers: int = 1, layer_contents: List[int] = None, dropout: bool = True,
                 activation_function: torch.nn.Module = torch.nn.Sigmoid):
        super().__init__()
        self.n_param = n_param
        self.n_out = n_out
        self.n_layers = n_layers
        self.layer_contents = ([self.n_param] * self.n_layers) if layer_contents is None else layer_contents
        self.layers = torch.nn.ModuleList()
        self.activation = activation_function
        self.initialized_activation = self.activation()
        self.initialize_layers()
        self.dropout = torch.nn.Dropout(0.5)
        self.do_dropout = dropout

    def initialize_layers(self):
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(self.n_param, self.layer_contents[0]))
        for i in range(self.n_layers - 1):
            self.layers.append(torch.nn.Linear(self.layer_contents[i],
                                               self.layer_contents[i + 1]))
        self.layers.append(torch.nn.Linear(self.layer_contents[-1], self.n_out))
        self.initialized_activation = self.activation()

    def forward(self, x):
        current = x
        for layer in self.layers[:-1]:
            current = layer(current)
            if self.do_dropout:
                current = self.dropout(current)
            # print(current.shape)
            current = self.initialized_activation(current)
        current = self.layers[-1](current)
        # print(current.shape)
        # current = self.initialized_activation(current)
        # print(current.shape)
        return current


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    lerr = 0
    lcount = 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collects Percent Error
        y = y.float()
        perr = ((y - pred).abs() / y) * 100
        perr = perr[~(perr.isinf() | perr.isnan())]
        terr = round(float((sum(perr) / len(perr)).cpu().detach().numpy()), 3)
        lerr += terr
        lcount += 1

    #         if batch % 100 == 0:
    #             loss, current = loss.item(), batch * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return lerr / lcount


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    lerr = 0
    lcount = 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X.float())
            #             print(pred.shape)
            #             print(pred)
            test_loss += loss_fn(pred, y.float()).item()

            y = y.float()
            perr = ((y - pred).abs() / y) * 100
            perr = perr[~(perr.isinf() | perr.isnan())]
            terr = round(float((sum(perr) / len(perr)).cpu().detach().numpy()), 3)
            lerr += terr
            lcount += 1

    test_loss = lerr / lcount
    print(f"Test Error: \nAvg loss: {test_loss:>8f} \n")
    return test_loss
