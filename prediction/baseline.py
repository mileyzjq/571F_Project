# baseline of passing distribution predictor based on average of passing networks in group stage for 2014-15 season

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
        allGames = ["matchday" + str(i) for i in range(1, 7)]
        for i in allGames:
            path = folder + i + "/networks/"
            for j in os.listdir(path):
                if re.search("-edges", j):
                    teamName = classes.getTeamNameFromNetwork(j)
                    with open(path + j) as file:
                        lines = file.readlines()
                        for line in lines:
                            p1, p2, weight = line.rstrip().split("\t")
                            p_key = p1 + "-" + p2
                            allPasses[teamName][p_key] += float(weight) / 6
                            totalTeamPasses[teamName] += float(weight)

        # calculate average loss
        avgLoss = 0
        count = 0
        path = folder + "r-16/networks/"
        for i in os.listdir(path):
            if re.search("-edges", i):
                teamName = classes.getTeamNameFromNetwork(i)
                with open(path + i) as file:
                    lines = file.readlines()
                    for line in lines:
                        p1, p2, weight = line.rstrip().split("\t")
                        p_key = p1 + "-" + p2
                        avgPass = allPasses[teamName][p_key]
                        loss = (avgPass - float(weight)) ** 2
                        avgLoss += loss
                        count += 1
        print(("Baseline Average Loss: {}".format(avgLoss / count)))


base = Baseline()
base.predict()
