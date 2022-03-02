# baseline of passing distribution predictor based on average of passing networks in group stage for 2014-15 season

import os
import re
import classes
from collections import defaultdict

folder = "../data/passing_distributions/2014-15/"
allPasses = defaultdict(lambda: defaultdict(float))
totalTeamPasses = defaultdict(float)

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
        print ("Baseline Average Loss: {}".format(avgLoss / count))

base = Baseline()
base.predict()