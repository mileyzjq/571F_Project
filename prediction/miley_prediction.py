import classes
import copy
import os
import re
import random
import math
import numpy as np
from collections import defaultdict
import util


class PredictPD():

    def __init__(self):
        self.learning_rate = 0.0092
        self.momentum = 0.09

        self.weights = defaultdict(float)
        self.delta_weights = defaultdict(float)
        self.matches = defaultdict(str)
        self.pass_between_postion = defaultdict(lambda: defaultdict(int))
        self.pass_between_players = defaultdict(lambda: defaultdict(int))
        self.total_pass = defaultdict(int)
        self.team_position = defaultdict(lambda: defaultdict(str))
        self.team_pass_complete = defaultdict(int)
        self.team_pass_attempt = defaultdict(int)
        self.team_pass_perc = defaultdict(float)
        self.team_stats = defaultdict(lambda: defaultdict(list))

        # Initialize directory paths
        self.pd_dir = "../data/passing_distributions/2014-15/"
        counts_dir = "../data/counts/avg_passes_count.txt"
        squad_dir = "../data/squads/2014-15/squad_list/"
        rank_dir = "../data/rankings/2013_14_rankings.txt"
        game_pos_dir = "../data/games_by_pos/perTeam/"
        self.save_file_dir = "../data/processed/player_data2.csv"

        self.matchday = ["matchday" + str(i) for i in range(1, 7)]
        # self.matchday.append("r-16")
        # self.matchday.append("q-finals")

        # Initialize features
        self.count_avg_pass_feature = classes.CountAvgPassesFeature(counts_dir)
        self.player_position_feature = classes.PlayerPositionFeature(squad_dir)
        self.rank_feature = classes.RankingFeature(rank_dir)
        self.mean_degree_feature = classes.MeanDegreeFeature()
        self.betweeness_feature = classes.BetweennessFeature()
        self.pass_attempt_feature = classes.PassesComplAttempPerPlayerFeature()
        self.pass_position_feature = classes.CountPassesPerPosFeature(game_pos_dir, "group")
        self.team_pass_attempt_feature = classes.CountPassesComplAttempPerTeamFeature("group")
        self.init_team_postion(squad_dir)

        # Average pairwise error over all players in a team
    # given prediction and gold
    def evaluate(self, features, weight):
        score = self.calculate_score(features, self.weights)
        loss = (score - float(weight)) ** 2
        return (score, loss)

    # score is dot product of features & weights
    def calculate_score(self, features, weights):
        score = 0.0
        for v in features:
            score += float(features[v]) * float(weights[v])
        return score

    def calculate_gradient_loss(self, features, weights, label):
        scalar = 2 * self.calculate_score(features, weights) - label
        mult = copy.deepcopy(features)
        for f in mult:
            mult[f] = float(mult[f])
            mult[f] *= scalar
        return mult

    # use SGD to update weights
    def update_weights(self, features, weights, label):
        grad = self.calculate_gradient_loss(features, weights, label)
        for w in self.weights:
            self.delta_weights[w] = self.learning_rate * grad[w] + self.delta_weights[w] * self.momentum
            self.weights[w] -= self.delta_weights[w]

    def init_team_postion(self, squad_dir):
        for team in os.listdir(squad_dir):
            if re.search("-squad", team):
                teamFile = open(squad_dir + team, "r")
                team_name = re.sub("-squad.*", "", team)
                team_name = re.sub("_", " ", team_name)
                for player in teamFile:
                    num, name, position = player.rstrip().split(", ")
                    self.team_position[team_name][num] = position

    def get_rival_team(self, matchID, team_name):
        (team1, team2) = self.matches[matchID].split("/")
        if team1 == team_name:
            return team2
        else:
            return team1

    def get_matchday(self, matchID):
        matchID = int(matchID)
        if matchID <= 2014322:
            return 0
        elif matchID >= 2014323 and matchID <= 2014338:
            return 1
        elif matchID >= 2014339 and matchID <= 2014354:
            return 2
        elif matchID >= 2014355 and matchID <= 2014370:
            return 3
        elif matchID >= 2014371 and matchID <= 2014386:
            return 4
        elif matchID >= 2014387 and matchID <= 2014402:
            return 5
        elif matchID >= 2014403 and matchID <= 2014418:
            return 6
        elif matchID >= 2014419 and matchID <= 2014426:
            return 7
        elif matchID >= 2014427 and matchID <= 2014430:
            return 8

    def extract_feature(self, team_name, p1, p2, matchID, weight):
        features = defaultdict(float)
        features["avg_pass"] = self.count_avg_pass_feature.getCount(team_name, p1, p2)
        features["check_same_postion"] = self.player_position_feature.isSamePos(team_name, p1, p2)
        rival_team = self.get_rival_team(matchID, team_name)
        features["check_diff_rank"] = self.rank_feature.isHigherInRank(team_name, rival_team)

        position1 = self.team_position[team_name][p1]
        position2 = self.team_position[team_name][p2]
        p_key = position1 + "-" + position2
        self.pass_between_postion[team_name][p_key] += int(weight)
        self.total_pass[team_name] += int(weight)
        features["avg_pass_position"] = self.pass_between_postion[team_name][p_key] / float(self.total_pass[team_name])
        features["mean_degree"] = self.mean_degree_feature.getMeanDegree(matchID, team_name)
        features["between_P1"] = self.betweeness_feature.getBetweenCentr(matchID, team_name, p1)
        features["avg_pass_percentage_P1"] = self.pass_attempt_feature.getPCPerc(team_name, p1)
        return features

    # store match data for all games, including team and opponent team
    def initialize_match(self):
        matches = copy.deepcopy(self.matchday)
        if "r-16" not in matches:
            matches.append("r-16")
        if "q-finals" not in matches:
            matches.append("q-finals")
        if "s-finals" not in matches:
            matches.append("s-finals")

        for matchday in matches:
            path = self.pd_dir + matchday + "/networks/"
            for network in os.listdir(path):
                if re.search("-edges", network):
                    team_name = classes.getTeamNameFromNetwork(network)
                    matchID = re.sub("_.*", "", network)

                    m = self.matches[matchID]
                    if m == "":
                        self.matches[matchID] = team_name
                    else:
                        self.matches[matchID] += "/" + team_name

        all_scores = open("../data/scores/2014-15_allScores.txt", "r")
        self.matchesWithScores = [line.rstrip() for line in all_scores]
        self.teamPlayedWith = defaultdict(list)
        self.teamWonAgainst = defaultdict(list)

        # for every team, store opponents in order by matchday
        for match in self.matchesWithScores:
            team1, score1, score2, team2 = match.split(", ")
            team1Won = 0
            if score1 > score2:
                team1Won = 1

            self.teamPlayedWith[team1].append(team2)
            self.teamPlayedWith[team2].append(team1)
            self.teamWonAgainst[team1].append(team1Won)
            self.teamWonAgainst[team2].append(abs(1 - team1Won))

    def initialize_team_status(self):
        for matchday in self.matchday:
            path = self.pd_dir + matchday + "/networks/"
            # iterate over games
            for network in os.listdir(path):
                if re.search("-team", network):
                    team_name = classes.getTeamNameFromNetwork(network)
                    team_name = re.sub("-team", "", team_name)
                    matchID = re.sub("_.*", "", network)

                    stats_file = open(path + network, "r")
                    for line in stats_file:
                        stats = line.rstrip().split(", ")

                    self.team_stats[team_name][matchID] = stats

    # Training
    # 	have features calculate numbers based on data
    # 	learn weights for features via supervised data (group stage games) and SGD/EM
    def train(self):
        iterations = 1
        self.initialize_match()
        self.initialize_team_status()
        output_list = []

        for i in range(iterations):
            avg_loss = 0
            pass_count = 0
            # for w in self.weights:
            # 	print ("weights[{}] = {}".format(w, float(self.weights[w])))
            match_count = 0
            games = []

            for matchday in self.matchday:
                path = self.pd_dir + matchday + "/networks/"
                for network in os.listdir(path):
                    if re.search("-edges", network):
                        games.append((path, network))

            for game in games:
                path, network = game
                edge_file = open(path + network, "r")
                team_name = classes.getTeamNameFromNetwork(network)
                matchID = re.sub("_.*", "", network)
                for players in edge_file:
                    p1, p2, target = players.rstrip().split("\t")
                    features = self.extract_feature(team_name, p1, p2, matchID, target)
                    score, loss = self.evaluate(features, target)
                    self.update_weights(features, self.weights, float(target))

                    features["p1"] = p1
                    features["p2"] = p2
                    features["target"] = target
                    features["check_diff_rank"] = 1 if features["check_diff_rank"] else 0
                    output_list.append(features)
                    avg_loss += loss
                    pass_count += 1
                match_count += 1
            util.toCSV(output_list, self.save_file_dir)
            print("Save {} entry".format(len(output_list)))
            print("Match {} - Average loss: {}".format(i, avg_loss / pass_count))

    # Testing
    #	Predict, then compare with dev/test set (r-16 games)
    def test(self):
        # sum up average error
        print("------- Testing -------")
        avg_loss = 0
        pass_count = 0
        match_count = 0
        matchday = "r-16"
        path = self.pd_dir + matchday + "/networks/"
        # matchday = "q-finals"
        # matchday = "s-finals"

        for network in os.listdir(path):
            if re.search("-edges", network):
                edge_file = open(path + network, "r")
                predict_edge_file = open("../predicted/pred-" + network, "w+")
                team_name = classes.getTeamNameFromNetwork(network)
                matchID = re.sub("_.*", "", network)

                for players in edge_file:
                    p1, p2, weight = players.rstrip().split("\t")
                    # print ("p1: {}, p2: {}, weight: {}".format(p1, p2, float(weight)))

                    features = self.extract_feature(team_name, p1, p2, matchID, weight)
                    # for f in features:
                    # 	print ("features[{}] = {}".format(f, float(features[f])))
                    for w in self.weights:
                        print(("weights[{}] = {}").format(w, float(self.weights[w])))

                    score, loss = self.evaluate(features, weight)
                    predict_edge_file.write(p1 + "\t" + p2 + "\t" + str(score) + "\t" + str(loss) + "\n")
                    avg_loss += loss
                    pass_count += 1
                match_count += 1

        print("Average loss: {}".format(avg_loss / pass_count))
        print(("Total average loss: {}").format(avg_loss))
        print(("Total examples: {}").format(pass_count))

pred = PredictPD()
pred.train()
pred.test()
#pred.train_Neural_Net()
#pred.test_neural_net()

# ------------------- extract feature function

# features["avg_pass_position"] = self.pass_position_feature.getCount(team_name, p_key)
# avgPassCompl = self.team_pass_attempt_feature.getPCCount(team_name, match_count)
# avgPassAttem = self.team_pass_attempt_feature.getPACount(team_name, match_count)
# features["avg_pass_percentage_P1"] = self.team_pass_attempt_feature.getPCPerc(team_name, match_count)
# avgPassFail = avgPassCompl - avgPassAttem
#
# oppAvgPassCompl = self.team_pass_attempt_feature.getPCCount(rival_team, match_count)
# oppAvgPassAttem = self.team_pass_attempt_feature.getPACount(rival_team, match_count)
# oppAvgPassPerc = self.team_pass_attempt_feature.getPCPerc(rival_team, match_count)
# oppAvgPassFail = oppAvgPassCompl - oppAvgPassAttem
#
# # for feature: won against a similar ranking team
# # 1. define history that we are able to use, i.e. previous games
# history = self.teamPlayedWith[team_name][:matchday]

# if len(history) > 0:
#     def computeSim(rank1, rank2):
#         return (rank1 ** 2 + rank2 ** 2) ** 0.5
#
#     # 2. find most similar opponent in terms of rank
#     # TODO: similarity could be defined better?
#     oppTeamRank = self.rank_feature.getRank(rival_team)
#     simTeam = ""
#     simTeamDistance = float('inf')
#     rank1 = oppTeamRank
#     for team in history:
#         rank2 = self.rank_feature.getRank(team)
#         sim = computeSim(rank1, rank2)
#         if sim < simTeamDistance:
#             simTeamDistance = sim
#             simTeam = sim
# 3. find out whether the game was won or lost
# features["wonAgainstSimTeam"] = self.teamWonAgainst[team_name][matchday]

# features["betwPerGameP2"] = self.betweeness_feature.getBetweenCentr(matchID, team_name, p2)

# features["avgPassComplPerP1"] = self.pass_attempt_feature.getPC(team_name, p1)
# features["avgPassComplPerP2"] = self.pass_attempt_feature.getPC(team_name, p2)
# features["avgPassAttempPerP1"] = self.pass_attempt_feature.getPA(team_name, p1)
# features["avgPassAttempPerP2"] = self.pass_attempt_feature.getPA(team_name, p2)
# features["avgPCPercPerP1"] = self.pass_attempt_feature.getPCPerc(team_name, p1)
# features["avgPCPercPerP2"] = self.pass_attempt_feature.getPCPerc(team_name, p2)
