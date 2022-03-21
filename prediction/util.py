import _pickle as pickle
import csv


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

