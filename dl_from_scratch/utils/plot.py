import matplotlib.pyplot as plt

def acc_loss_plot(accuracies, losses, figname="plot.png"):
    x = list(range(1,len(accuracies)+1))

    fig, ax1 = plt.subplots()
    ax1.plot(x, accuracies,label="Accuracy")

    ax2 = ax1.twinx()
    ax2.plot(x, losses,label="Loss", color="orange")

    plt.savefig(f"{figname}")

