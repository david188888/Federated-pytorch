from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from xie.task import Net, get_weights
from typing import List, Tuple, Dict, Optional
from logging import INFO
import flwr as fl
import time

total_round = 0
training_losses = []
training_times = []
evaluation_accuracy = []

def fit_metrics_aggregator(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    global num_rounds, total_round
    total_round += 1
    losses = [metric["loss"] for _, metric in metrics]
    times = [metric["time"] for _, metric in metrics]
    
    avg_loss = sum(losses) / len(losses)
    avg_time = sum(times) / len(times)

    training_losses.append(avg_loss)
    training_times.append(avg_time)
    if total_round == num_rounds:
        name = "idd, {} client, {} round".format(len(metrics), num_rounds)
        with open('training_log.txt', 'a') as f:
            f.write(name + '\n')
            f.write(str(training_losses))
            f.write('\n')
            f.write(str(training_times))
            f.write('\n')
    
    
    return {"loss": avg_loss}

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute weighted average of metrics."""
    # Multiply accuracy of each client by number of examples used
    global num_rounds, total_round
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if total_round == num_rounds:
        name = "non-idd, {} client, {} round".format(len(metrics), num_rounds)
        with open('training_log.txt', 'a') as f:
            f.write(str(evaluation_accuracy))
            f.write('\n')

    evaluation_accuracy.append(sum(accuracies) / sum(examples))

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples)
    }


def server_fn(context: Context) -> ServerAppComponents:
    # Read from config
    global num_rounds
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn= fit_metrics_aggregator,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
