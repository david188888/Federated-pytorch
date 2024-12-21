# xie: A Flower / PyTorch app

## Install dependencies and project




```bash
pip install -r requirements.txt
pip install -e .
```

## Run with the Simulation Engine

In the `xie` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

if you want to try some other model with different client amount and sampling strategy

you can:

- modify the pyproject.toml file, you can change the client amout, training epoch, batch_size
- modify the client_app.py, find the `client_fn` and change the `load_data_iid` and `load_data_non_iid` to choose your sampling strategy

## Run the centralized one

in `xie/xie` directory, run task file

```python 
python task.py
```

make sure the path of the dataset is correct

## Output

the `training_log.txt` file saves the output and performance of the models with different client amount and dataset sampling strategy and also the centralized one.

the `training_log.png`is the line plot that summarize the performance of all trained models.



# Project Architecture

## task.py

### Classes

- Net

  : Defines the neural network model.

### Functions

- get_weights(net): Returns the weights of the model as a list of numpy arrays.
- set_weights(net, parameters): Sets the weights of the model from a list of numpy arrays.
- load_data_non_iid(partition_id, num_partitions, batch_size): Loads non-IID partitioned data.
- load_data_iid(partition_id, num_partitions, batch_size): Loads IID partitioned data.
- train(net, trainloader, epoch): Trains the model for a specified number of epochs.
- test(net, testloader): Tests the model and returns the loss and accuracy.
- train_centralize(net, trainloader, epoch): Trains the model centrally and logs the loss, time, and accuracy.

### Main Execution

- Loads the MNIST dataset.
- Trains the model centrally.
- Logs the training results.

## client_app.py

- FlowerClient

  : Defines the Flower client.

  - __init__: Initializes the client with the model, datasets, and training parameters.
  - fit: Trains the model and returns the updated weights and metrics.
  - evaluate: Evaluates the model and returns the loss, accuracy, and metrics.

### Functions

- client_fn(context): Initializes and returns a FlowerClient instance.

### Main Execution

- Defines the ClientApp with the client_fn.

## server_app.py

### Global Variables

- total_round: Tracks the total number of rounds.
- training_losses, training_times, evaluation_accuracy: Lists for logging training metrics.

### Functions

- fit_metrics_aggregator(metrics): Aggregates and logs training metrics.
- weighted_average(metrics): Computes the weighted average of evaluation metrics.
- server_fn(context): Initializes and returns ServerAppComponents.

### Main Execution

- Defines the ServerApp with the server_fn.



