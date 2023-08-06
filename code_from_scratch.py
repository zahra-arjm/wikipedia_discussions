#import needed libraries
import numpy as np
import pandas as pd
import json
with open("frequency_data.json", "r") as f:
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
conversation_type = ['esc', 'non-esc']

transition_dict = {}
for method in methods:
    transition_dict[method] = {}
    for type in conversation_type:
        transition_dict[method][type] = df