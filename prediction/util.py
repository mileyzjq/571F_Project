import _pickle as pickle
import csv
import matplotlib.pyplot as plt


def getFeatureSet():
    key_sets = dict()

    key_sets["All"] = ["avg_pass", "check_same_postion", "check_diff_rank", "between_P1", "between_P2",
                       "avg_pass_percentage_P1", "avg_pass_percentage_P2"]

    key_sets["Avg Passes"] = ["check_same_postion", "check_diff_rank", "between_P1", "between_P2",
                              "avg_pass_percentage_P1", "avg_pass_percentage_P2"]

    key_sets["Same Pos."] = ["avg_pass", "check_diff_rank", "between_P1", "between_P2",
                             "avg_pass_percentage_P1", "avg_pass_percentage_P2"]

    key_sets["Avg Passes/player"] = ["avg_pass", "check_same_postion", "check_diff_rank", "between_P1", "between_P2"]

    key_sets["Betweeness"] = ["avg_pass", "check_same_postion", "check_diff_rank", "avg_pass_percentage_P1", "avg_pass_percentage_P2"]

    return key_sets


def plot_multiple_res(epoch, res_set, train_result_path):
    for name, result in res_set.items():
        val_loss = result["val"]
        epochs = range(epoch)
        plt.plot(epochs, val_loss, label=name)

    plt.title("Loss Plot")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(train_result_path)


def toFig(loss_rec, saved_path, added_name=""):
    train_loss = loss_rec["train"]
    val_loss = loss_rec["val"]
    epoch = len(train_loss)
    plt.plot(range(epoch), train_loss, label="train loss")
    plt.plot(range(epoch), val_loss, label="val loss")
    plt.title("{} Loss".format(added_name))
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.legend(loc='upper right')
    plt.savefig(saved_path)


def fromCSV(saved_path):
    with open(saved_path) as f:
        output_csv = [{k: float(v) for k, v in row.items()}
                      for row in csv.DictReader(f, skipinitialspace=True)]
    return output_csv


def toCSV(toSave, path):
    """

    :param toSave: list of dictionary, toSave[0] is the first data record
    :param path:
    :return:
    """
    with open(path, 'w', encoding='utf8', newline='') as output_file:
        fc = csv.DictWriter(output_file, fieldnames=toSave[0].keys(), )
        fc.writeheader()
        fc.writerows(toSave)


def toTxt(s, path):
    with open(path, "a+") as text_file:
        text_file.write(s)


def fromTxt(path):
    with open(path) as file:
        return file.readlines()


def toPickle(obj, path):
    with open(path, 'wb+') as handle:
        pickle.dump(obj, handle)


def fromPickle(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
