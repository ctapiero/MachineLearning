# Author Cristian Tapiero Machine Learning CS6350

import numpy as np
import pandas as pd

#global data
df = pd.read_csv('data/dummy.csv')

class DecisionTreeNode:
    def __init__(self,branches=None,label=None,entropy=None,name=None):
        self.name = name
        self.branches = branches           # Dictionary mapping feature values to child nodes
        self.label = label  # Predicted label for leaf nodes
        self.entropy = entropy

# this function builds a simple Descision Tree based on the ID3 algorithm 
# It uses entropy for choosing the root node


def entropy(p_pass,p_fails,set):
    if(p_pass == 0 or p_fails == 0): return 0
    return - ((p_pass/set)*np.log2(p_pass/set)) - ((p_fails/set)*np.log2(p_fails/set))

# this fuction calculates the best info gain of each attribute in the dataset      
def calculate_BestInfoGain(labels,df): #Atribute is the column of the data frame
    
    df.insert(1, "Label", labels)
    # label = "Ripe"
    #calculate the Information gain for each attribute
    infoGain_values = {}
    for (attribute, set)  in df.iteritems():
        
        if(attribute == "Label"): continue
        features = df[attribute].unique()
        # calculating the number of pass and fail labels to compute entropy for each value of Attribute
        average = 0
        s = len(df)
        # keep track of attribute with highest gain
        
        for x in features:
            p_pass = df[(df[attribute] == x) & (df["Label"] == "T")]
            p_pass = len(p_pass)
            p_fails = df[(df[attribute] == x) & (df["Label"] == "F")]
            p_fails = len(p_fails)

            sv = p_pass + p_fails
            Hs = entropy(p_pass,p_fails,sv)
            average += Hs * (p_pass + p_fails) / s
        
        infoGain = 1 - average
        infoGain_values[attribute] = infoGain
    
    #getting the attribute with the highest gain
    best_attribute = max(infoGain_values, key=lambda k: infoGain_values[k])
    best_gain = infoGain_values[best_attribute]
    return best_attribute,best_gain

def ID3(S,attributes):

   
    label_column = 'Ripe'
    attributes = df.drop(columns=label_column)
    labels= df[label_column]
    #recursion base condition
    unique_values = np.unique(labels)
    unique_len = len(unique_values)

   # all labels are the same therefore return node with label
    if(unique_len == 1): 
        return DecisionTreeNode(label=unique_values[0])

    # else create note with atributes and split base on entropy 
    best_attribute,best_gain = calculate_BestInfoGain(labels,attributes)
    print(best_attribute,best_gain)
    rootNode = DecisionTreeNode(name=best_attribute,entropy=best_gain)
    

def main():
    ID3(df,df)
    
if __name__ == "__main__":
    main()
