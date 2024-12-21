from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays
from xie.task import Net, get_weights, load_data_non_iid, set_weights, test, train, load_data_iid
import torch
from typing import Dict, List, Tuple
import time


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
        start_time = time.time()
        loss = train(
            net=self.net,
            trainloader=self.train_dataset,
            epoch=self.local_epochs,
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Loss: {loss}")
        return get_weights(self.net), len(self.train_dataset), {"loss": float(loss), "time": training_time}
    
    
    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        """Return evaluation loss, number of samples, and a dict with additional metrics."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, testloader=self.test_dataset)
        samples = len(self.test_dataset)
        metrics = {"accuracy": float(accuracy)}
        print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return float(loss), samples, metrics

    
def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data_iid(partition_id, num_partitions,batch_size)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

