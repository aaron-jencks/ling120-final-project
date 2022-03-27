from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
import torch
from typing import List

# logger = set_logger('running_data_boosting_classifier', use_tb_logger=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('.\\elapsed_output.csv').dropna()
print(list(df.columns))
df.head()
features = ['latitude', 'longitude', 'elevation', 'elapsed_time', 'distance_travelled', 'cumulative_distance', 'average_speed']
X = df[features]
y = preprocessing.LabelEncoder().fit_transform(df['pace_skewed'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


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
        current = self.initialized_activation(current)
        # print(current.shape)
        return current


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #         if batch % 100 == 0:
    #             loss, current = loss.item(), batch * len(X)
    #             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss.item()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            pred = model(X)
            #             print(pred.shape)
            #             print(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


output_count = np.unique(y)
print(output_count)
tensor_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train.to_numpy().astype(np.single)).to(device),
                                              torch.tensor(y_train, dtype=torch.long).to(device))
tensor_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test.to_numpy().astype(np.single)).to(device),
                                             torch.tensor(y_test.astype(np.int), dtype=torch.long).to(device))
tensor_train.classes = output_count
tensor_test.classes = output_count


def test_model(layers: int, size: int, dropout: bool, epochs: int = 10) -> float:
    # Version 1 (No Gradient Boosting)
    model = GeneralPerceptron(len(features), len(output_count), layers, [size] * layers, dropout).to(device)
    # Version 2 (Gradient Boosting)
    # model = GradientBoostingClassifier(model, 10, cuda=torch.cuda.is_available())
    # model.set_optimizer('SGD', lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss = train_loop(torch.utils.data.DataLoader(tensor_train, batch_size=256), model, criterion, optimizer)
    i = 0
    while loss > 1 and i < epochs:
        print('{}x{}: Training iteration {}, Loss {}'.format(layers, size, i, loss))
        loss = train_loop(torch.utils.data.DataLoader(tensor_train, batch_size=256), model, criterion, optimizer)
        print('Training Error: {}'.format(loss))
        i += 1

    return test_loop(torch.utils.data.DataLoader(tensor_test, batch_size=256), model, criterion)



