from pyvis.network import Network
from itertools import compress
import numpy as np

# a function to pick one of the lables if there are multiple labels available for a part of an utter
def pick_label(list_of_labels, super_dict_labels, method):
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


# a function to give the respective short labels 
# given the long labels
def long_ver_to_short_label(label, labels_joint):
  for (short, long) in labels_joint:
    if label == long:
      return short


# a function to update transition matrices based on the threshold 
# (get both esc and non-esc) matrices to keep their label consistent
def update_transition_matrices(transition_matrices, threshold_in, conversation_type):

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
  for c_type in conversation_type:
    transition_matrices[c_type]['Other'] , transition_matrices[c_type].loc['Other'] = \
      transition_matrices[c_type][transition_matrix_sum.columns[filter]].sum(axis=1), \
      transition_matrices[c_type].loc[transition_matrix_sum.columns[filter]].sum(axis=0)
  # see if 'Other' survives now
  filter['Other'] = transition_matrix_sum['Other'].sum(axis=0) < threshold_in

  for c_type in conversation_type:
    transition_matrices[c_type].drop(transition_matrix_sum.columns[filter],
                          axis=0,
                          inplace=True)
    transition_matrices[c_type].drop(transition_matrix_sum.columns[filter],
                          axis=1,
                          inplace=True)
  return transition_matrices 


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



def plot_network(transition_matrix, c_type):

  net = Network(height='900px', width='900px',directed =True)
  # a threshold for showing the edge
  threshold = 12
  node_color = '#0000CC'
  if c_type == 'esc':
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
               x=int(X[idx_n]), y=int(Y[idx_n]),
               color=node_color)
  # add edges
  for idx_s, source in enumerate(sorted_DH + sorted_non_DH):
    for idx_t, target in enumerate(sorted_DH + sorted_non_DH):
      #remove transitions to the source itself
      if target == source:
        continue

      edge_weight = int(transition_matrix[target][source])
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
  return net
