import matplotlib.pyplot as plt


def line_plot(title, x, ys, labels, x_label, y_label, save_path):
    plt.close()
    plots = [plt.plot(x, y, label=label)[0] for y, label in zip(ys, labels)]
    plt.legend(handles=plots)
    save_plot(save_path, title, x_label, y_label)


def save_plot(save_path, title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path, bbox_inches='tight')
