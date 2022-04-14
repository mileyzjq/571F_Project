from collections import defaultdict
import os
import re
import snap


def get_match_id(network):
    return re.sub("_.*", "", network)


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


# average passes completed percentage feature
class PassesComplAttempPerPlayerFeature():
    def __init__(self):
        forder_list = get_network_file_list(False, "-players", "+")
        self.pass_completed_per_player = defaultdict(lambda: defaultdict(float))
        self.pass_attempted_per_player = defaultdict(lambda: defaultdict(float))
        self.pass_completed_percent = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            team_name = get_team_name(network)
            team_name = re.sub("-players", "", team_name)

            playerFile = open(path + network, "r")
            players = [line.rstrip() for line in playerFile]
            for player in players:
                id, pc, pa, percPc = player.split(",")
                self.pass_completed_per_player[team_name][id] += float(pc) / 6.0
                self.pass_attempted_per_player[team_name][id] += float(pa) / 6.0
                self.pass_completed_percent[team_name][id] += float(percPc) / 6.0

    def getPCPerc(self, team_name, num):
        return self.pass_completed_percent[team_name][num]


class CountAvgPassesFeature():
    def __init__(self):
        counts_dir = "../data/counts/avg_passes_count.txt"
        self.avg_count = defaultdict(lambda: defaultdict(float))
        count_file = open(counts_dir, "r")

        for line in count_file:
            team, players, weight = line.strip().split(", ")
            self.avg_count[team][players] = weight

    def getCount(self, team, player1, player2):
        p_key = player1 + "-" + player2
        return self.avg_count[team][p_key]


class PlayerPositionFeature():
    def __init__(self):
        squad_dir = "../data/squads/2014-15/squad_list/"
        self.team_player_name = defaultdict(lambda: defaultdict(str))
        self.team_player_pos = defaultdict(lambda: defaultdict(str))

        def get_team_name_from_squad(team_file):
            team_name = re.sub("-squad.*", "", team_file)
            return re.sub("_", " ", team_name)

        for team in os.listdir(squad_dir):
            if re.search("-squad", team):
                team_file = open(squad_dir + team, "r")
                team_name = get_team_name_from_squad(team)
                for player in team_file:
                    num, name, pos = player.rstrip().split(", ")
                    self.team_player_name[team_name][num] = name
                    self.team_player_pos[team_name][num] = pos

    def getPos(self, teamName, num):
        return self.team_player_pos[teamName][num]

    def isSamePos(self, team_name, num1, num2):
        if self.getPos(team_name, num1) != self.getPos(team_name, num2):
            return 0
        return 1


# compare the rank of team A and team B
class RankingFeature():
    def __init__(self):
        rank_dir = "../data/rankings/2013_14_rankings.txt"
        self.rank = defaultdict(int)
        rank_file = open(rank_dir, "r")
        for rank in rank_file:
            rank, team = rank.rstrip().split(", ")
            self.rank[team] = int(rank)

    def getRank(self, team):
        return self.rank[team]

    def isHigherInRank(self, team1, team2):
        return self.getRank(team1) > self.getRank(team2)


# average degree of team edges
class MeanDegreeFeature():
    def __init__(self):
        forder_list = get_network_file_list(True, "-edges")
        self.mean_drgree = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            edge_file = open(path + network, "r")
            degree_per_player = defaultdict(int)
            team_name = get_team_name(network)
            match_ID = get_match_id(network)

            for players in edge_file:
                p1, p2, weight = players.rstrip().split("\t")
                degree_per_player[p1] += 1

            node_file = open(path + match_ID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
            players = [line.rstrip() for line in node_file]
            player_count = len(players)
            total_degree = 0
            for player in degree_per_player:
                total_degree += degree_per_player[player]

            avg_degree = total_degree / player_count
            self.mean_drgree[match_ID][team_name] = avg_degree

    def getMeanDegree(self, match_ID, team_name):
        return self.mean_drgree[match_ID][team_name]


class CountPassesComplAttempPerTeamFeature():
    def __init__(self):
        forder_list = get_network_file_list(False, "-team")
        self.pass_compl_percent_team = defaultdict(float)

        for (path, network) in forder_list:
            team_file = open(path + network, "r")
            team_name = get_team_name(network)
            team_name = re.sub("-team", "", team_name)

            for line in team_file:
                stats = line.rstrip().split(", ")
                self.pass_compl_percent_team[team_name] += float(stats[2])

    def getPCPerc(self, team_name, matchNum):
        return self.pass_compl_percent_team[team_name] / 6.0


def generateGraph(path, network):
    edge_file = open(path + network, "r")
    team_name = get_team_name(network)
    match_ID = get_match_id(network)
    edges = [line.rstrip() for line in edge_file]
    node_file = open(path + match_ID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
    players = [line.rstrip() for line in node_file]
    # generate graph
    graph = snap.TNGraph.New()
    for player in players:
        num, name = player.split("\t")
        graph.AddNode(int(num))
    for edge in edges:
        src, dest, weight = edge.split("\t")
        graph.AddEdge(int(src), int(dest))
    return graph


# Returns the average betweenness centrality of each player
# calculated only using group stage, like average degree
class BetweennessFeature():
    def __init__(self):
        forder_list = get_network_file_list(False, "-edges")
        self.betweeness_centrality = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            graph = generateGraph(path, network)
            team_name = get_team_name(network)
            Nodes, Edges = graph.GetBetweennessCentr(1.0)
            players = [(node, Nodes[node]) for node in Nodes]
            for player in players:
                num, betweeness = player
                self.betweeness_centrality[team_name][num] += betweeness

        # normalize
        for team_name in self.betweeness_centrality:
            for num in self.betweeness_centrality[team_name]:
                self.betweeness_centrality[team_name][num] /= 6

    def getBetweenCentr(self, team_name, player):
        return self.betweeness_centrality[team_name][int(player)]


# get GNN closeness centrality feature
class closenessFeature():
    def __init__(self):
        forder_list = get_network_file_list(False, "-edges")
        self.closeness_centrality = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            graph = generateGraph(path, network)
            team_name = get_team_name(network)
            # get closeness centrality for each player
            for NI in graph.Nodes():
                CloseCentr = graph.GetClosenessCentr(NI.GetId())
                # print("node: %d centrality: %f" % (NI.GetId(), CloseCentr))
                self.closeness_centrality[team_name][NI.GetId()] += CloseCentr

        # normalize
        for team_name in self.closeness_centrality:
            for num in self.closeness_centrality[team_name]:
                self.closeness_centrality[team_name][num] /= 6

    def get_closeness(self, team_name, player):
        return self.closeness_centrality[team_name][int(player)]


# get GNN page rank centrality feature
class pageRankFeature():
    def __init__(self):
        forder_list = get_network_file_list(False, "-edges")
        self.page_rank = defaultdict(lambda: defaultdict(float))

        for (path, network) in forder_list:
            graph = generateGraph(path, network)
            team_name = get_team_name(network)
            # get page rank centrality for each player
            PRankH = graph.GetPageRank()
            for item in PRankH:
                print(item, PRankH[item])
                self.page_rank[team_name][item] += PRankH[item]

        for team_name in self.page_rank:
            for num in self.page_rank[team_name]:
                self.page_rank[team_name][num] /= 6

    def get_page_rank(self, team_name, player):
        return self.page_rank[team_name][int(player)]
