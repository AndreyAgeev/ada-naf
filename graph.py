import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_file(file_name):
    data = {}
    current_model = ''
    dataset_name = ''
    with open(file_name, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if "reqularization" in line:
            current_model = line.strip().split(":")[0]
            if "NAF-1-LAYER" in current_model:
                current_model = "ADA-NAF-1"
            elif "NAF-3-LAYER" in current_model:
                current_model = "ADA-NAF-3"
            elif "NAF-MH-3-HEAD-1-LAYER" in current_model:
                current_model = "ADA-MH-3-NAF-1"
            data[current_model] = {}
        elif "dataset" in line:
            # Extract avg_auc and std_auc values
            parts = line.split(',')
            avg_auc = float(parts[0].split(':')[-1].strip())
            std_auc = float(parts[1].split(':')[-1].strip())
            data[current_model][dataset_name]['avg_auc'].append(avg_auc)
            data[current_model][dataset_name]['std_auc'].append(std_auc)
        elif "logger" not in line:
            dataset_name = line.strip()
            data[current_model][dataset_name] = {'avg_auc': [], 'std_auc': []}
    return data


def plot_and_save_data(data, filename='combined_graphs.png'):
    proportions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # proportions = [0.0, 0.05, 0.1]

    datasets = set()
    for model_data in data.values():
        datasets.update(model_data.keys())

    n_graphs = len(datasets)
    n_rows = (n_graphs + 1) // 2
    is_odd = n_graphs % 2 != 0

    fig = plt.figure(figsize=(16, 6 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig)

    for i, dataset in enumerate(sorted(datasets)):
        if is_odd and i == n_graphs - 1:
            ax = fig.add_subplot(gs[i // 2, :])
        else:
            row, col = divmod(i, 2)
            ax = fig.add_subplot(gs[row, col])

        for model, model_data in data.items():
            if dataset in model_data:
                avg_aucs, std_aucs = model_data[dataset]["avg_auc"], model_data[dataset]["std_auc"]
                ax.errorbar(proportions, avg_aucs, yerr=std_aucs, fmt='-o', capsize=5, label=model)

        ax.set_xlabel('Proportions')
        ax.set_ylabel('Average AUC')
        ax.set_title(f'Average AUC for {dataset.capitalize()} dataset')
        ax.set_xticks(proportions)
        # ax.set_ylim([-0.1, 1.1])
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    file_path = "/Users/andreyageev/PycharmProjects/NAF/output_num_seeds_3_num_cross_val_3_num_trees_100_count_epoch_50_contaminations_0_20240208_211658.txt"
    parsed_data = parse_file(file_path)
    plot_and_save_data(parsed_data, filename="check.png")
