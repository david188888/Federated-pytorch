from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from xie.task import Net, get_weights, load_data, set_weights, test, train
import torch




# Define Flower client
class FlowerClient(NumPyClient):
    
    def __init__(self,net, train_dataset, test_dataset,local_epochs):
        self.net = Net()    
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
    
    
    
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        result = train(
            net = self.net,
            trainloader = self.train_dataset,
            epoch = self.local_epochs,
            testloader = self.test_dataset
        )
        return get_weights(self.net), len(self.train_dataset), result
    
    
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, testloader=self.test_dataset)
        return loss, len(self.test_dataset), {'accuracy': accuracy}

    
def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions,batch_size)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
  
