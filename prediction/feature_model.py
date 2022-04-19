import features as Feature
import os
import re
from collections import defaultdict
import util


class FeatureModel:
    def __init__(self):
        self.learning_rate = 0.0092
        self.momentum = 0.09
        self.pd_dir = "../data/passing_distributions/2014-15/"
        squad_dir = "../data/squads/2014-15/squad_list/"
        self.save_file_dir = "../data/processed/player_data4.csv"
        self.matchday = ["matchday" + str(i) for i in range(1, 7)]
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

        # Initialize features
        self.count_avg_pass_feature = Feature.CountAveragePassFeature()
        self.player_position_feature = Feature.PlayerPositionsFeature()
        self.rank_feature = Feature.RankFeature()
        self.mean_degree_feature = Feature.MeanDegreeFeature()
        self.betweeness_feature = Feature.BetweennessCentralFeature()
        self.closeness_feature = Feature.ClosenessFeature()
        self.page_rank_feature = Feature.PageRankFeature()
        self.pass_attempt_feature = Feature.PassCompletedPlayerFeature()
        self.team_pass_completed_feature = Feature.PassCompletedTeamFeature()
        self.init_team_postion(squad_dir)

    def init_team_postion(self, squad_dir):
        for team in os.listdir(squad_dir):
            if re.search("-squad", team):
                team_file = open(squad_dir + team, "r")
                team_name = re.sub("-squad.*", "", team)
                team_name = re.sub("_", " ", team_name)
                for player in team_file:
                    num, name, position = player.rstrip().split(", ")
                    self.team_position[team_name][num] = position

    def get_rival_team(self, match_ID, team_name):
        (team1, team2) = self.matches[match_ID].split("/")
        if team1 == team_name:
            return team2
        return team1

    def extract_feature(self, team_name, p1, p2, match_ID, weight):
        features = defaultdict(float)
        features["avg_pass"] = self.count_avg_pass_feature.get_count(team_name, p1, p2)
        features["check_same_postion"] = self.player_position_feature.check_same_pos(team_name, p1, p2)
        rival_team = self.get_rival_team(match_ID, team_name)
        features["check_diff_rank"] = self.rank_feature.check_higher_rank(team_name, rival_team)

        position1 = self.team_position[team_name][p1]
        position2 = self.team_position[team_name][p2]
        p_key = position1 + "-" + position2
        self.pass_between_postion[team_name][p_key] += int(weight)
        self.total_pass[team_name] += int(weight)
        features["avg_pass_position"] = self.pass_between_postion[team_name][p_key] / float(self.total_pass[team_name])
        features["mean_degree"] = self.mean_degree_feature.get_mean_degree(match_ID, team_name)
        features["between_P1"] = self.betweeness_feature.get_betweeness(team_name, p1)
        features["between_P2"] = self.betweeness_feature.get_betweeness(team_name, p2)
        features["closeness_P1"] = self.closeness_feature.get_closeness(team_name, p1)
        features["closeness_P2"] = self.closeness_feature.get_closeness(team_name, p2)
        features["page_rank_P1"] = self.page_rank_feature.get_page_rank(team_name, p1)
        features["page_rank_P2"] = self.page_rank_feature.get_page_rank(team_name, p2)
        features["avg_pass_percentage_P1"] = self.pass_attempt_feature.get_perc_completed_player(team_name, p1)
        features["avg_pass_percentage_P2"] = self.pass_attempt_feature.get_perc_completed_player(team_name, p2)
        features["pass_compl_percent_team"] = self.team_pass_completed_feature.get_team_perc_completed(team_name)
        return features

    # get match data for all teams
    def initialize_match(self):
        forder_list = Feature.get_network_file_list(False, "-edges")

        for (path, network) in forder_list:
            team_name = Feature.get_team_name(network)
            match_ID = re.sub("_.*", "", network)
            if self.matches[match_ID] == "":
                self.matches[match_ID] = team_name
            else:
                self.matches[match_ID] += "/" + team_name

    # Training
    def train(self):
        self.initialize_match()
        forder_list = Feature.get_network_file_list(False, "-edges")
        output_list = []
        match_count = 0

        for (path, network) in forder_list:
            edge_file = open(path + network, "r")
            team_name = Feature.get_team_name(network)
            match_ID = re.sub("_.*", "", network)
            for players in edge_file:
                p1, p2, target = players.rstrip().split("\t")
                features = self.extract_feature(team_name, p1, p2, match_ID, target)
                features["p1"] = p1
                features["p2"] = p2
                features["target"] = target
                features["check_diff_rank"] = 1 if features["check_diff_rank"] else 0
                output_list.append(features)
            match_count += 1
        util.toCSV(output_list, self.save_file_dir)
        print("Save {} entry".format(len(output_list)))


pred = FeatureModel()
pred.train()
