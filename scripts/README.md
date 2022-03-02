# A network-assisted approach to predicting passing distributions #
## Final project for CS224W at Stanford University ##
###Autumn 2015###

In this directory are a number of scripts used to calculate counts used for
features, plot bar graphs, and calculate general statistics.

* `count_avg_passes_feat.py`: Accumulates average number of passes during the
  group stage for each player pair for each team
* `count_pass_per_pos.py`: Accumulates counts of how often certain positions
  pass to each other based on passing distributions [group stage, semifinals]
* `count_spec_passes_feat.py`: Accumulates exact number of passes during the
  group stage for each player pair for each team
* `count_tot_edges_nodes.py`: Counts total number of edges, nodes, and average
  counts for entire 2014-15 season
* `plot_games_by_pos.py`: Plots bar graphs for opposing teams in matches to
  visualize number of passes between positions
* `verify_PC_assump.py`: Calculates the percentage times that a team with a
  higher passsing completion rate and higher passing volume is also the winning
  team during the 2014-15 season.
