import re
import matplotlib.pyplot as plt

def parse_training_log(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sections = re.split(r'\n(?=\w)', content)
    data = []

    for section in sections:
        lines = section.strip().split('\n')
        if len(lines) < 4:
            continue

        title = lines[0]
        loss = eval(lines[1])
        training_time = eval(lines[2])
        acc = eval(lines[3])

        data.append({
            'title': title,
            'loss': loss,
            'training_time': training_time,
            'acc': acc
        })

    return data

def plot_data(data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    for entry in data:
        title = entry['title']
        loss = entry['loss']
        acc = entry['acc']

        rounds = list(range(1, len(loss) + 1))

        if 'non-iid' in title.lower():
            linestyle = '--'
        else:
            linestyle = '-'

        axs[0].plot(rounds, loss, label=title, linestyle=linestyle)
        axs[1].plot(rounds, acc, label=title, linestyle=linestyle)

    axs[0].set_title('Loss over Rounds')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].set_title('Accuracy over Rounds')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('distributed-system/xie/training_log_plot.png')
    plt.show()

if __name__ == "__main__":
    file_path = 'distributed-system/xie/training_log.txt'
    data = parse_training_log(file_path)
    plot_data(data)