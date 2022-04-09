from collections import defaultdict
import os
import re
import snap

def getMatchIDFromFile(network):
	match_ID = re.sub("_.*", "", network)
	return match_ID

def getTeamNameFromNetwork(network):
	team_name = re.sub("[^-]*-", "", network, count=1)
	team_name = re.sub("-edges", "", team_name)
	team_name = re.sub("_", " ", team_name)
	return team_name

def get_network_file_list(folder, is_append, keyword):
	all_games = ["matchday" + str(i) for i in range(1, 7)]
	list = []
	if is_append:
		all_games.append("r-16")
		all_games.append("q-finals")
		all_games.append("s-finals")

	for game in all_games:
		path = folder + game + "/networks/"
		for network in os.listdir(path):
			if re.search(keyword, network):
				list.append((path, network))
	return list

def get_network_file_list_remove(path, is_append, keyword, avoid):
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
			for network in os.listdir(path):
				if avoid not in network:
					if re.search(keyword, network):
						list.append((path, network))
	return list



class CountAvgPassesFeature():
	def __init__(self, count_file_name):
		self.avg_counts = defaultdict(lambda: defaultdict(float))
		count_file = open(count_file_name, "r")
		for line in count_file:
			team, players, weight = line.strip().split(", ")
			self.avg_counts[team][players] = weight

	def getCount(self, team, player1, player2):
		p_key = player1 + "-" + player2
		return self.avg_counts[team][p_key]

class PlayerPositionFeature():
	def __init__(self, squad_dir):

		def getTeamNameFromFile(teamFile):
			team_name = re.sub("-squad.*", "", teamFile)
			team_name = re.sub("_", " ", team_name)
			return team_name

		self.team_num_name = defaultdict(lambda: defaultdict(str))
		self.team_num_pos = defaultdict(lambda: defaultdict(str))

		for team in os.listdir(squad_dir):
			if re.search("-squad", team):
				path = squad_dir + team
				teamFile = open(squad_dir + team, "r")
				team_name = getTeamNameFromFile(team)
				for player in teamFile:
					num, name, pos = player.rstrip().split(", ")
					self.team_num_name[team_name][num] = name
					self.team_num_pos[team_name][num] = pos

	def getPos(self, team_name, num):
		return self.team_num_pos[team_name][num]

	def getName(self, team_name, num):
		return self.team_num_name[team_name][num]

	def isSamePos(self, team_name, num1, num2):
		ret = 1
		if self.getPos(team_name, num1) != self.getPos(team_name, num2):
			ret = 0
		return ret

class RankingFeature():
	def __init__(self, rankFileName):
		self.rankings = defaultdict(int)
		rank_file = open(rankFileName, "r")
		for rank in rank_file:
			rank, team = rank.rstrip().split(", ")
			self.rankings[team] = int(rank)

	def getRank(self, team):
		return self.rankings[team]

	def isHigherInRank(self, team1, team2):
		return self.getRank(team1) > self.getRank(team2)

	def getDiffInRank(self, team1, team2):
		return self.getRank(team1) - self.getRank(team2)

# unsuccessful feature
class MeanDegreeFeature():
	def __init__(self):
		self.meanDegree = defaultdict(lambda: defaultdict(float))
		meanDegree_path_list = get_network_file_list("../data/passing_distributions/2014-15/", True, "-edges");

		for path_pair in meanDegree_path_list:
			edge_file = open(path_pair[0] + path_pair[1], "r")
			degree_per_player = defaultdict(int)
			team_name = getTeamNameFromNetwork(path_pair[1])
			match_ID = getMatchIDFromFile(path_pair[1])
			# print "team: %s" % team_name
			totalDegree = 0

			for players in edge_file:
				p1, p2, weight = players.rstrip().split("\t")
				# print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))
				degree_per_player[p1] += 1
			
			# count number of nodes to take average over team
			nodeFile = open(path_pair[0] + match_ID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
			players = [line.rstrip() for line in nodeFile]
			numPlayers = len(players)
			totalDegree = 0
			for player in degree_per_player:
				totalDegree += degree_per_player[player]

			avgDegree = totalDegree / numPlayers
			# print "Avg degree for %s is %f" % (team_name, avgDegree)
			self.meanDegree[match_ID][team_name] = avgDegree
	
	def getMeanDegree(self, match_ID, team_name):
		return self.meanDegree[match_ID][team_name]

# Returns the average betweenness centrality of each player
# calculated only using group stage, like average degree
class BetweennessFeature():
	def __init__(self):
		self.betweenCentr = defaultdict(lambda: defaultdict(float))
		betweenness_path_list = get_network_file_list("../data/passing_distributions/2014-15/", False, "-edges");
		
		for path_pair in betweenness_path_list:
			edge_file = open(path_pair[0] + path_pair[1], "r")

			degree_per_player = defaultdict(int)
			team_name = getTeamNameFromNetwork(path_pair[1])
			match_ID = getMatchIDFromFile(path_pair[1])

			edges = [line.rstrip() for line in edge_file]

			nodeFile = open(path_pair[0] + match_ID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
			players = [line.rstrip() for line in nodeFile]

			# build each network
			PlayerGraph = snap.TNGraph.New()
			for player in players:
				num, name = player.split("\t")
				PlayerGraph.AddNode(int(num))
			for edge in edges:
				src, dest, weight = edge.split("\t")
				src = int(src)
				dest = int(dest)
				PlayerGraph.AddEdge(src, dest)

			# calculate betweenness
			Nodes = snap.TIntFltH()
			Edges = snap.TIntPrFltH()
			snap.GetBetweennessCentr(PlayerGraph, Nodes, Edges, 1.0)

			players_by_between = [(node, Nodes[node]) for node in Nodes]
			for player in players_by_between:
				num, betw = player
				self.betweenCentr[team_name][num] += betw

		# normalize over number of matchdays
		for team_name in self.betweenCentr:
			for num in self.betweenCentr[team_name]:
				self.betweenCentr[team_name][num] /= 6

	def getBetweenCentr(self, match_ID, team_name, player):
		return self.betweenCentr[team_name][int(player)]

# average passes completed and attempted per player feature
# averaged over all group games
class PassesComplAttempPerPlayerFeature():
	def __init__(self):
		passesComp_path_list = get_network_file_list_remove("../data/passing_distributions/2014-15/", False, "-player", "+");

		self.pc_per_player = defaultdict(lambda: defaultdict(float))
		self.paPerPlayer = defaultdict(lambda: defaultdict(float))
		self.pcPercPerPlayer = defaultdict(lambda: defaultdict(float))

		for path_pair in passesComp_path_list:
			playerFile = open(path_pair[0] + path_pair[1], "r")
			team_name = getTeamNameFromNetwork(path_pair[1])
			team_name = re.sub("-players", "", team_name)
			match_ID = getMatchIDFromFile(path_pair[1])

			players = [line.rstrip() for line in playerFile]
			for player in players:
				num, pc, pa, percPc = player.split(",")
				self.pc_per_player[team_name][num] += float(pc) / 6.0
				# print "team_name: %s, num: %s, %f" % (team_name, num, self.pc_per_player[team_name][num])
				self.paPerPlayer[team_name][num] += float(pa) / 6.0
				self.pcPercPerPlayer[team_name][num] += float(percPc) / 6.0


	def getPC(self, team_name, num):
		return self.pc_per_player[team_name][num]

	def getPA(self, team_name, num):
		return self.pc_per_player[team_name][num]

	def getPCPerc(self, team_name, num):
		return self.pcPercPerPlayer[team_name][num]

# pre-load passes by position by match_ID
class CountPassesPerPosFeature():
	def __init__(self, count_file_dir, train_end):
		self.countsByPos = defaultdict(lambda: defaultdict(float))

		folders = []
		if train_end == "group":
			folders.append("group/")
		elif train_end == "r-16":
			folders.append("group/")
			folders.append("r-16/")
		elif train_end == "q-finals":
			folders.append("group/")
			folders.append("r-16/")
			folders.append("q-finals/")

		# total passes per team
		self.totalCounts = defaultdict(float)
		for stage in folders:
			path = count_file_dir + stage
			for teamByGame in os.listdir(path):
				if ".DS_Store" not in teamByGame:
					teamGameFile = open(path + teamByGame, "r")
					# get team_name from filename
					team_name = re.sub(".*-", "", teamByGame)
					team_name = re.sub("_", " ", team_name)
					for line in teamGameFile:
						pos, weight = line.rstrip().split("\t")
						self.countsByPos[team_name][pos] += float(weight)
						self.totalCounts[team_name] += float(weight)

		for team_name in self.countsByPos:
			for pos in self.countsByPos[team_name]:
				self.countsByPos[team_name][pos] /= self.totalCounts[team_name]


	def getCount(self, team, pos):
		return self.countsByPos[team][pos]

# pre-load passes completed/attempted
class CountPassesComplAttempPerTeamFeature():
	def __init__(self, train_end):

		self.passComplPerTeam = defaultdict(int)
		self.passAttemPerTeam = defaultdict(int)
		self.passPercPerTeam = defaultdict(float)

		folder = "../data/passing_distributions/2014-15/"
		if train_end == "r-16":
			folders.append("r-16/")
		elif train_end == "q-finals":
			folders.append("r-16/")
			folders.append("q-finals/")

		countPassesComp = get_network_file_list(folder, False, "-team")

		for path_pair in countPassesComp:
			teamFile = open(path_pair[0] + path_pair[1], "r")
			team_name = getTeamNameFromNetwork(path_pair[1])
			team_name = re.sub("-team", "", path_pair[1])
			match_ID = getMatchIDFromFile(path_pair[1])

			for line in teamFile:
				stats = line.rstrip().split(", ")
				self.passComplPerTeam[team_name] += float(stats[0])
				self.passAttemPerTeam[team_name] += float(stats[1])
				self.passPercPerTeam[team_name] += float(stats[2])
		
	def getPCCount(self, team_name, matchNum):
		return self.passComplPerTeam[team_name] / (matchNum + 1.0)

	def getPACount(self, team_name, matchNum):
		return self.passAttemPerTeam[team_name] / (matchNum + 1.0)

	def getPCPerc(self, team_name, matchNum):
		return self.passPercPerTeam[team_name] / (matchNum + 1.0)

	def getPassFail(self, team_name, matchNum):
		return self.getPCCount(self, team_name, matchNum) - self.getPACount(self, team_name, matchNum)
