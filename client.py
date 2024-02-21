from typing import Dict, Tuple
from flwr.common import NDArrays
from centralized import train, test, load_data, load_model
from collections import OrderedDict
import torch
import flwr as fl


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k:torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict)  


net = load_model()
train_dataset, test_dataset = load_data()

# Define Flower client
class FLoerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, train_dataset, 1)
        return self.get_parameters(config), len(train_dataset), {}
    
    
    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, test_dataset)
        return loss, len(test_dataset), {'accuracy': accuracy}
    
    
fl.client.start_numpy_client(
    server_address='127.0.0.1:8086',
    client = FLoerClient(),
)
        
