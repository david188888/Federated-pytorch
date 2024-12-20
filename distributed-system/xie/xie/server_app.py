from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from xie.task import Net, get_weights
from typing import List, Tuple, Dict, Optional
from logging import INFO
import flwr as fl

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import time

total_round = 1
training_losses = []
training_times = []

def fit_metrics_aggregator(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    global num_rounds, total_round
    losses = [metric["loss"] for _, metric in metrics]
    times = [metric["time"] for _, metric in metrics]
    
    avg_loss = sum(losses) / len(losses)
    avg_time = sum(times) / len(times)

    training_losses.append(avg_loss)
    training_times.append(avg_time)
    if total_round == num_rounds:
        plot_training_loss()
        plot_training_time()
    total_round += 1
    
    return {"loss": avg_loss}

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute weighted average of metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples)
    }

def plot_training_loss():
    plt.figure()
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title('Training Loss per Round')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()


def plot_training_time():
    plt.figure()
    plt.plot(training_times, label='Training Time')
    plt.xlabel('Round')
    plt.ylabel('Time (s)')
    plt.title('Training Time per Round')
    plt.legend()
    plt.savefig('training_time.png')
    plt.close()


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


class MyServerApp(ServerApp):
    def on_fit_round_end(self, round_number, results, failures):
        # 在每一轮结束时记录损失和训练时间
        # 假设每个结果中有损失信息，并且你可以从结果中提取训练时间
        losses = [result.metrics["loss"] for result in results]
        times = [result.metrics["time"] for result in results]

        avg_loss = sum(losses) / len(losses)
        avg_time = sum(times) / len(times)
        
        training_losses.append(avg_loss)  # 记录每轮的平均损失
        training_times.append(avg_time)    # 记录每轮的平均训练时间
        
        # 如果是最后一轮，可以绘制图表
        if round_number == self.config["num_rounds"] - 1:
            print("images are saved!")
            plot_training_loss()
            plot_training_time()



# Create ServerApp
app = MyServerApp(server_fn=server_fn)
