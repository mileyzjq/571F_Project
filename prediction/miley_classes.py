from collections import defaultdict
import os
import re
import snap
import numpy as np
import pandas as pd

def get_match_id(network):
	return re.sub("_.*", "", network)

def getTeamNameFromNetwork(network):
	team_name = re.sub("[^-]*-", "", network, count=1)
	team_name = re.sub("-edges", "", team_name)
	return re.sub("_", " ", team_name)

def get_network_file_list(is_append, keyword, avoid_word):
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


# average passes completed and attempted per player feature
# averaged over all group games
class PassesComplAttempPerPlayerFeature():
	def __init__(self):
		forder_list = get_network_file_list(False, "-players", "+")
		print(forder_list)
		self.pcPerPlayer = defaultdict(lambda: defaultdict(float))
		self.paPerPlayer = defaultdict(lambda: defaultdict(float))
		self.pcPercPerPlayer = defaultdict(lambda: defaultdict(float))

		for (path, network) in forder_list:
			team_name = getTeamNameFromNetwork(network)
			team_name = re.sub("-players", "", team_name)

			# file = pd.read_csv(path+network, header=None)
			# #players = [line.rstrip() for line in file]
			# for index, row in file.iterrows():
			# 	match, pc, pa, percPc = row
			playerFile = open(path + network, "r")
			players = [line.rstrip() for line in playerFile]
			for player in players:
				num, pc, pa, percPc = player.split(",")
				self.pcPerPlayer[team_name][num] += float(pc) / 6.0
				# print "teamName: %s, num: %s, %f" % (teamName, num, self.pcPerPlayer[teamName][num])
				self.paPerPlayer[team_name][num] += float(pa) / 6.0
				self.pcPercPerPlayer[team_name][num] += float(percPc) / 6.0
		print(self.pcPercPerPlayer)

	def getPC(self, team_name, num):
		return self.pcPerPlayer[team_name][num]

	def getPA(self, team_name, num):
		return self.pcPerPlayer[team_name][num]

	def getPCPerc(self, team_name, num):
		return self.pcPercPerPlayer[team_name][num]

class CountAvgPassesFeature():
	def __init__(self, count_file_name):
		self.avgCounts = defaultdict(lambda: defaultdict(float))
		count_file = open(count_file_name, "r")
		for line in count_file:
			team, players, weight = line.strip().split(", ")
			self.avgCounts[team][players] = weight

	def getCount(self, team, player1, player2):
		p_key = player1 + "-" + player2
		return self.avgCounts[team][p_key]

class PlayerPositionFeature():
	def __init__(self, squad_dir):

		def getteam_nameFromFile(teamFile):
			team_name = re.sub("-squad.*", "", teamFile)
			team_name = re.sub("_", " ", team_name)
			return team_name

		self.teamNumName = defaultdict(lambda: defaultdict(str))
		self.teamNumPos = defaultdict(lambda: defaultdict(str))

		for team in os.listdir(squad_dir):
			if re.search("-squad", team):
				path = squad_dir + team
				teamFile = open(squad_dir + team, "r")
				team_name = getteam_nameFromFile(team)
				for player in teamFile:
					num, name, pos = player.rstrip().split(", ")
					self.teamNumName[team_name][num] = name
					self.teamNumPos[team_name][num] = pos

	def getPos(self, team_name, num):
		return self.teamNumPos[team_name][num]

	def getName(self, team_name, num):
		return self.teamNumName[team_name][num]

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
		folder = "../data/passing_distributions/2014-15/"
		allGames = ["matchday" + str(i) for i in range(1, 7)]
		allGames.append("r-16")
		allGames.append("q-finals")
		allGames.append("s-finals")

		self.meanDegree = defaultdict(lambda: defaultdict(float))

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-edges", network):
					edgeFile = open(path + network, "r")

					degreePerPlayer = defaultdict(int)
					team_name = getTeamNameFromNetwork(network)
					matchID = get_match_id(network)
					# print "team: %s" % team_name
					totalDegree = 0

					for players in edgeFile:
						p1, p2, weight = players.rstrip().split("\t")
						# print "p1: %s, p2: %s, weight: %f" % (p1, p2, float(weight))
						degreePerPlayer[p1] += 1

					# count number of nodes to take average over team
					nodeFile = open(path + matchID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
					players = [line.rstrip() for line in nodeFile]
					numPlayers = len(players)
					totalDegree = 0
					for player in degreePerPlayer:
						totalDegree += degreePerPlayer[player]

					avgDegree = totalDegree / numPlayers
					# print "Avg degree for %s is %f" % (team_name, avgDegree)
					self.meanDegree[matchID][team_name] = avgDegree
	
	def getMeanDegree(self, matchID, team_name):
		return self.meanDegree[matchID][team_name]

# Returns the average betweenness centrality of each player
# calculated only using group stage, like average degree
class BetweennessFeature():
	def __init__(self):
		folder = "../data/passing_distributions/2014-15/"
		allGames = ["matchday" + str(i) for i in range(1, 7)]
		# allGames.append("r-16")
		# allGames.append("q-finals")
		# allGames.append("s-finals")

		self.betweenCentr = defaultdict(lambda: defaultdict(float))

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-edges", network):
					edgeFile = open(path + network, "r")

					degreePerPlayer = defaultdict(int)
					team_name = getTeamNameFromNetwork(network)
					matchID = get_match_id(network)

					edges = [line.rstrip() for line in edgeFile]

					nodeFile = open(path + matchID + "_tpd-" + re.sub(" ", "_", team_name) + "-nodes", "r")
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

	def getBetweenCentr(self, matchID, team_name, player):
		return self.betweenCentr[team_name][int(player)]

# pre-load passes by position by matchID
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

		allGames = ["matchday" + str(i) for i in range(1, 7)]

		if train_end == "r-16":
			folders.append("r-16/")
		elif train_end == "q-finals":
			folders.append("r-16/")
			folders.append("q-finals/")

		# allGames.append("r-16")
		# allGames.append("q-finals")
		# allGames.append("s-finals")

		for matchday in allGames:
			path = folder + matchday + "/networks/"
			for network in os.listdir(path):
				if re.search("-team", network):
					teamFile = open(path + network, "r")
					team_name = getTeamNameFromNetwork(network)
					team_name = re.sub("-team", "", team_name)
					matchID = get_match_id(network)
					# print "team_name is: %s" % team_name
					# print "matchID is: %s" % matchID
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
