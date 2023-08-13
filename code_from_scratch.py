#import needed libraries
import numpy as np
import pandas as pd
import json
from itertools import compress

with open("docs/frequency_data.json", "r") as f:
  raw_data = json.load(f)
# install pyvis
# pyvis is a network visualization library based on Networkx
#which allow graphs to be interactive. You can also save them as html
# !pip install pyvis
from pyvis.network import Network

from collections import Counter
super_dict_labels = {}
for i in range(len(raw_data)):


  dict_list = ([dict(Counter(x)) for x in raw_data[i]['utt_labels']])
  for item in dict_list:
    for k, v in item.items():
      if k in super_dict_labels.keys():
        super_dict_labels[k] += v
      else:
        super_dict_labels[k] = v

# change label "DH-1: Bailing out" to formal format "DH1: Bailing out"
for conversation in raw_data:
  for labels in conversation['utt_labels']:
    #check if DH- is there
    if "DH-1: Bailing out" in labels:
      idx = labels.index("DH-1: Bailing out")
      labels[idx] = "DH1: Bailing out"

#count frequency of each label in a dictionary after changing DH-1 to DH1
super_dict_labels = {}
for i in range(len(raw_data)):


  dict_list = ([dict(Counter(x)) for x in raw_data[i]['utt_labels']])
  for item in dict_list:
    for k, v in item.items():
      if k in super_dict_labels.keys():
        super_dict_labels[k] += v
      else:
        super_dict_labels[k] = v

# a function to pick one of the lables if there are multiple labels available for a part of an utter
def pick_label(list_of_labels, method):
  '''
  This function picks one of the lables in case there are multiple labels
   available.
  In case both DH and non-DH labels are availabe it picks the DHs.
  In case all of the labels are non-DH it picks the one with the highest frequency.

  Args:
  list_of_labels (list): labels for part of the utterance
  method (string): if max or min, picks the highest or lowest DH. If all gives back all labels
  '''

  # first check no of labels or if method is 'all'
  if len(list_of_labels) == 1 or method == 'all':
    final_label = list_of_labels
  else: #case of multiple labels
    #check how many DHs are present
    count_DH = 0
    for item in list_of_labels:
      if item[0:2] == 'DH':
        count_DH += 1
    if count_DH == 0:
      #pick the one with the highest frequency
      label_freqs = [super_dict_labels[x] for x in list_of_labels]
      final_label = [list(super_dict_labels.keys())[list(super_dict_labels.values()).index(max(label_freqs))]]
    elif count_DH == 1:
      # pick the DH
      final_label = [i for i in list_of_labels if i.startswith('DH')]
    else:

      #extract DH_no s
      DH_no = [int(i[2]) for i in list_of_labels if i.startswith('DH')]
      # select the DH based on the method specified
      if method == 'min':
        final_label = [i for i in list_of_labels
                       if (i.startswith('DH') and int(i[2]) == min(DH_no))]
      elif method == 'max':
        final_label = [i for i in list_of_labels
                       if (i.startswith('DH') and int(i[2]) == max(DH_no))]

  return final_label


def long_ver_to_short_label(label):
  return labels_short[labels_long.index(label)]

# put the list of labels into labels_long variable
labels_long = list(super_dict_labels.keys())
#add 'end' label to the list of labels
labels_long.append('end')
# view long labels
print(labels_long)

#short labels
labels_short = ['Coords',
 'Context',
 'clarify',
 'DH5',
 'DH4:Repeat',
 'Suggest',
 'DH1:Ad',
 'DH4:stance',
 'Other',
 'DH0',
 'DH3',
 'Ask',
 'Conced',
 'DH6',
 'DH2',
 'DH1:Bail',
 'DH7',
 "Don't know",
 'Other:Quote',
 'DH5:arg with',
 "DH6:with evid",
 'DH1:Attack',
 'DH4:without evid',
 'DH2:off',
 'end']

# make a join data structure for short and long labels
# to handle and filter easier
labels_joint = [(short, long) for (short, long) in zip(labels_short, labels_long)]

# for the transitions, the data will be stored in a dictionary with
# each method as a key, and the values are another dict with 
# keys as escalated and non-escalated type and values equal to a 
# dataframe. The df is indexed based on the labels

# create empty dateframe with rows and columns labeled by 
# short labels and size equal to any empty transition matrix

transition_mtx = \
    np.zeros(
        (len(labels_short), len(labels_short))
        , float)
df = pd.DataFrame(data=transition_mtx,
                  columns=labels_short,
                  index=labels_short)


#methods to be applied
methods = ['all', 'max', 'min']
conversation_type = ['non-esc', 'esc']

transition_dict = {}
for method in methods:
    transition_dict[method] = {}
    for type in conversation_type:
        transition_dict[method][type] = df.copy()

# fill out the transition matrices
for conversation in raw_data:
  # used idx because we need to access to the next utterance 
  # in the same for loop
  for idx_utter in range(len(conversation['utt_labels'])):

    #loop through methods
    for method in methods:
      # find from and to labels
      from_labels = pick_label(conversation['utt_labels'][idx_utter], method)
      len_from = len(from_labels)

      # two for loops in case we wanted to handle more than 1 label:
      for from_label in from_labels:
        #find the short form of the label
        from_label = long_ver_to_short_label(from_label)

        if idx_utter == len(conversation['utt_labels'])-1:
          #if it's the last part of conversation the to_label is 'end'
          to_labels = ['end']
          
        else:
          to_labels = pick_label(conversation['utt_labels'][idx_utter+1], method)
        
        len_to = len(to_labels)
        for to_label in to_labels:
          #find the short form of the label
          to_label = long_ver_to_short_label(to_label)
          transition_dict[method][conversation_type[conversation['escalation']]].loc[from_label,to_label] += \
            1/(len_from*len_to)

"""Multiple categories of the same DH are among labels. Since we need to simplify the most, I merged them!
Then, I created another filtered list of the labels and updated the transition matrix accordingly.
"""

# sum instances of similar DHs
rep_DHs = ['DH1', 'DH2', 'DH4', 'DH5', 'DH6']
to_be_removed = []
to_be_renamed = []
for DH in rep_DHs:
  
  #find DH instances
  DH_x_instances = [inst for inst in labels_short if inst.startswith(DH)]
  to_be_renamed.append(DH_x_instances[0])
  to_be_removed.extend(DH_x_instances[1:])
  # update the matrix by summing all instances into the first instance
  for method in methods:
    for type in conversation_type:
      transition_dict[method][type][DH_x_instances[0]], \
      transition_dict[method][type].loc[DH_x_instances[0]] = \
        transition_dict[method][type][DH_x_instances].sum(axis=1), \
        transition_dict[method][type].loc[DH_x_instances].sum()
      # rename the summed col and index
      transition_dict[method][type]. \
        rename(columns={DH_x_instances[0]:DH},
              index={DH_x_instances[0]:DH},
              inplace=True)
      # remove unwanted rows and cols
      transition_dict[method][type].drop(DH_x_instances[1:],
                                        axis=0,
                                        inplace=True)
      transition_dict[method][type].drop(DH_x_instances[1:],
                                        axis=1,
                                        inplace=True)
# update list of lables

# keep to be removed idx
to_be_removed_idx = []
for idx, short in enumerate(labels_short):
  # rename the first instance
  if short in to_be_renamed:
    # del labels_joint[idx]
    to_be_removed_idx.append(idx)
    labels_joint.append((short[:3], short[:3]))
  # remove other instances
  if short in to_be_removed:
    # del labels_joint[idx]
    to_be_removed_idx.append(idx)

# sort in reverse so it doesn't get messed up
for idx in sorted(to_be_removed_idx, reverse=True):
  del labels_joint[idx]


"""Let's see the total number of transitions in one of the methods (escalated and non-escalated conversations)"""

np.sum(np.sum(transition_dict[method][type], axis=0), axis=0)

"""We have ~2000 transition in mathod. To simplify, I merged labels with low inward transitions (because label 'end' does not have any outward transition).
After filtering, I gathered all the filtered labels into label 'Other'. Since there was an 'Other' label in the given labels, first, I checked if 'Other' label has survived filtering!
"""

# a function to update transition matrices based on the threshold 
# (get both esc and non-esc) matrices to keep their label consistent
def update_transition_matrices(transition_matrices, threshold_in):

  ''' This function gets the dict of the two transition matrics for the
  two type of the conversation (esc and non-esc) as an input and returns
  the curated matrices as output. 
  '''

  # sum both to apply the filter
  transition_matrix_sum = transition_matrices['esc'] + \
                          transition_matrices['non-esc']
  #find filtered labels
  filter = transition_matrix_sum.sum(axis=0) < threshold_in
  # sum all filtered cols and rows into 'Other'; to avoid
  # summing the 'Other' itselt twice, make the filter['Other']
  # true for now
  filter['Other'] = True
  for type in conversation_type:
    transition_matrices[type]['Other'] , transition_matrices[type].loc['Other'] = \
      transition_matrices[type][transition_matrix_sum.columns[filter]].sum(axis=1), \
      transition_matrices[type].loc[transition_matrix_sum.columns[filter]].sum(axis=0)
  # see if 'Other' survives now
  filter['Other'] = transition_matrix_sum['Other'].sum(axis=0) < threshold_in

  for type in conversation_type:
    transition_matrices[type].drop(transition_matrix_sum.columns[filter],
                          axis=0,
                          inplace=True)
    transition_matrices[type].drop(transition_matrix_sum.columns[filter],
                          axis=1,
                          inplace=True)
  return transition_matrices 

threshold_in = 20
# update transition_dict
for method in methods:
  transition_dict[method] = \
    update_transition_matrices(transition_dict[method], threshold_in)



"""## **Part 3: graph for filtered transitions and fixed nodes**

For the visualizing nodes, we wanted to divide the graph into two parts. A part for DH labels which are ordered based on their level. The other part would be all other lables. In the next cell, I separated the labels and also sorted them alphabetically.

"""






"""In the end, I built six (three, number of methods by two, escalated and non-escalated) graphs.
The size of each node is proportional to the number of inward transitions. The width of each edge is proportional to the number of transition between the two nodes (except for the transitions to node 'end').
I added a variable 'threshod' to omit the eadges with less number of transition. If any edge transition is lower than the 'threshod' the edge won't be shown in the graph.
I also have removed the transitions from an edge to itself. It helped declutter the graph substantially. Besides, it meant that the conversation remained in a state for a longer time (which we did not care about).

The graphs will be save as 'html' files in the current directory (after running next cell, open 'files' on the left menu bar; you should see two files with names "nonesc_fixed.html" and "esc_fixed.html"; the files are downloadable).

First, let's look at the 'max' graphs:
"""
def divide_labels_and_sort(transition_matrix):
  # get column names
  labels = transition_matrix.columns

  is_DH = [label.startswith('DH') for label in labels]
  sorted_DH = sorted(list(compress(labels, is_DH)), reverse=True)
  sorted_non_DH = sorted([label for label in labels if label not in sorted_DH])
  return sorted_DH, sorted_non_DH


def generate_node_coordinates(DH_labels, non_DH_labels):
  # empty list to keep the coordinates of the nodes
  ys = []
  xs = []
  # generate coordinates for all the nodes in the left part
  r = 300
  y = np.linspace(0, 2*r, len(DH_labels), dtype=int)
  x = (-np.sqrt(r**2 - ((y-r)**2)/1.1) - 50).astype(int)
  ys.extend(list(y))
  xs.extend(list(x))
  # generate coordinates for all the nodes in the right part
  y = np.linspace(0, 2*r, len(non_DH_labels), dtype=int)
  x = (np.sqrt(r**2 - ((y-r)**2)/1.1) + 50).astype(int)
  ys.extend(list(y))
  xs.extend(list(x))

  return xs, ys



def plot_network(transition_matrix, type):

  net = Network(height='900px', width='900px',directed =True)
  # a threshold for showing the edge
  threshold = 12
  node_color = '#0000CC'
  if type == 'esc':
    edge_color = '#FF9933'
  else:
    edge_color = '#33FF33'

  # add nodes to network
  #left column for DHs & right col for the rest
  # index is the sorted index in DH or non-DH (sotrted_DH or sorted_non_DH)
  # idx is the index in the filtered label matrix (all the labels)
  # each node would be part of an ellipse for the visualization purposes
  # each group of nodes is part of a half ellipse (two different circles)

  # get labels sorted and divide into DH and non-DH
  sorted_DH, sorted_non_DH = divide_labels_and_sort(transition_matrix)

  # get node coordinates
  X, Y = generate_node_coordinates(sorted_DH, sorted_non_DH)
  

  # add nodes
  for idx_n, node in enumerate(sorted_DH + sorted_non_DH):

    # get size of the node (frequency of inward transitions)
    node_size = int(transition_matrix[node].sum())
    net.add_node(idx_n, label=node, value=node_size*10,
               x= X[idx_n], y=Y[idx_n],
               color=node_color)
  # add edges
  for idx_s, source in enumerate(sorted_DH + sorted_non_DH):
    for idx_t, target in enumerate(sorted_DH + sorted_non_DH):
      #remove transitions to the source itself
      if target == source:
        continue

      edge_weight = int(transition_matrix[source][target])
      # show transitions to 'end' node anyway
      if target == 'end':
        edge_weight = threshold + 1

      if edge_weight > threshold:
        net.add_edge(idx_s, idx_t,
                  weight=edge_weight,
                  value=edge_weight,
                  color=edge_color)
  # toggle_physics method changes the position of nodes based on the strength
  # of the edges. Since we wanted to have the same positions for both graphs, this
  # option is off
  net.toggle_physics(False)
  net.show("bleh.html", notebook=False)

  # save/show the graph
  # return net
plot_network(transition_dict[method][type], type)

print('yay')