import miley_classes as classes
#import classes
import copy
import os
import re
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
        self.closeness_feature = classes.closenessFeature()
        self.page_rank_feature = classes.pageRankFeature()
        self.pass_attempt_feature = classes.PassesComplAttempPerPlayerFeature()
        self.pass_position_feature = classes.CountPassesPerPosFeature(game_pos_dir, "group")
        self.team_pass_attempt_feature = classes.CountPassesComplAttempPerTeamFeature("group")
        self.init_team_postion(squad_dir)

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
        features["between_P1"] = self.betweeness_feature.getBetweenCentr(team_name, p1)
        features["between_P2"] = self.betweeness_feature.getBetweenCentr(team_name, p2)
        features["closeness_P1"] = self.closeness_feature.get_closeness(team_name, p1)
        features["closeness_P2"] = self.closeness_feature.get_closeness(team_name, p2)
        features["page_rank_P1"] = self.page_rank_feature.get_page_rank(team_name, p1)
        features["page_rank_P2"] = self.page_rank_feature.get_page_rank(team_name, p2)
        features["avg_pass_percentage_P1"] = self.pass_attempt_feature.getPCPerc(team_name, p1)
        features["avg_pass_percentage_P2"] = self.pass_attempt_feature.getPCPerc(team_name, p2)
        #features["pass_pos_feature_p1"] = self.pass_position_feature.getCountPerc(team_name, self.team_position[team_name][p1])
        #features["pass_pos_feature_p2"] = self.pass_position_feature.getCountPerc(team_name,self.team_position[team_name][p2])
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
                    features["p1"] = p1
                    features["p2"] = p2
                    features["target"] = target
                    features["check_diff_rank"] = 1 if features["check_diff_rank"] else 0
                    output_list.append(features)
                match_count += 1
            util.toCSV(output_list, self.save_file_dir)
            print("Save {} entry".format(len(output_list)))

pred = PredictPD()
pred.train()
