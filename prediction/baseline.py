import os
import re
from collections import defaultdict

folder = "../data/passing_distributions/2014-15/"
allPasses = defaultdict(lambda: defaultdict(float))
totalTeamPasses = defaultdict(float)


def get_team_name(network):
    team_name = re.sub("[^-]*-", "", network, count=1)
    team_name = re.sub("-edges", "", team_name)
    return re.sub("_", " ", team_name)


def get_network_file_list(is_append, keyword, avoid_word="*&*+#"):
    folder = "../data/passing_distributions/2014-15/"
    all_games = ["matchday" + str(i) for i in range(1, 7)]
    list = []
    if is_append:
        all_games.append("r-16")
        all_games.append("q-finals")
        all_games.append("s-finals")

    for game in all_games:
        path = folder + game + "/networks/"
        for network in os.listdir(path):
            if avoid_word not in network and re.search(keyword, network):
                list.append((path, network))
    return list


# calculate averages
class Baseline():

    def predict(self):
        forder_list = get_network_file_list(False, "-edges")
        self.betweeness_centrality = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            team_name = get_team_name(network)
            with open(path + network) as file:
                lines = file.readlines()
                for line in lines:
                    p1, p2, target = line.rstrip().split("\t")
                    p_key = p1 + "-" + p2
                    allPasses[team_name][p_key] += float(target) / 6.0
                    totalTeamPasses[team_name] += float(target)

        # calculate average loss
        avgLoss = 0
        count = 0
        path = folder + "r-16/networks/"
        for i in os.listdir(path):
            if re.search("-edges", i):
                team_name = get_team_name(i)
                with open(path + i) as file:
                    lines = file.readlines()
                    for line in lines:
                        p1, p2, target = line.rstrip().split("\t")
                        p_key = p1 + "-" + p2
                        avgPass = allPasses[team_name][p_key]
                        loss = (avgPass - float(target)) ** 2
                        avgLoss += loss
                        count += 1
        print(("Baseline Average Loss: {}".format(avgLoss / count)))


base = Baseline()
base.predict()
