# Author Cristian Tapiero Machine Learning CS6350

import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(self):
        self.branches = {}           # Dictionary mapping feature values to child nodes
        self.label = ""  # Predicted label for leaf nodes
        self.entropy = 0

# this function builds a simple Descision Tree based on the ID3 algorithm 
# It uses entropy for choosing the root node
def ID3(S,attributes):
    calculate_infoGain(atribute,labels,Set)

def entropy(p_pass,p_fails,set):
    if(p_pass == 0 or p_fails == 0): return 0
    return - ((p_pass/set)*np.log2(p_pass/set)) - ((p_fails/set)*np.log2(p_fails/set))
       
def calculate_bestInfoGain(attribute,label,df): #Atribute is the column of the data frame

    # list all features
    features = df[attribute].unique()
    # calculating the number of pass and fail labels to compute entropy for each value of Attribute
    average = 0
    s = len(df)
    for x in features:
        p_pass = df[(df[attribute] == x) & (df[label] == "T")]
        p_pass = len(p_pass)
        p_fails = df[(df[attribute] == x) & (df[label] == "F")]
        p_fails = len(p_fails)

        sv = p_pass + p_fails
        Hs = entropy(p_pass,p_fails,sv)
        average += Hs * (p_pass + p_fails) / s
    
    gain = 1 - average
    print(gain)    
    return gain


def main():
    df = pd.read_csv('data/dummy.csv')
    calculate_bestInfoGain("Variety","Ripe",df)
    calculate_bestInfoGain("Color","Ripe",df)
    calculate_bestInfoGain("Smell","Ripe",df)
    calculate_bestInfoGain("Time","Ripe",df)

if __name__ == "__main__":
    main()
