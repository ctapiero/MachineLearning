# Author Cristian Tapiero Machine Learning CS6350

import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(self,branches=None,label=None,informationGain=None,name=None,feature=None):
        self.name = name #node name
        self.feature =feature # feature used to split data
        self.branches = branches # Dictionary mapping feature values to child nodes
        self.label = label  # Predicted label for leaf nodes
        self.informationGain = informationGain #entropy value at each node
    
    def depth(self):
        # Recursive function to calculate the depth of the tree
        if (self.name == "leaf"):  # Leaf node
            return 1
        else:
            # Depth is the maximum depth of child nodes + 1
            return 1 + max(node.depth() for node in self.branches.values())

# Example usage:
# this function builds a simple Descision Tree based on the ID3 algorithm 
# It uses entropy for choosing the root node


def entropy(p_pass,p_fails,set):
    if(p_pass == 0 or p_fails == 0): return 0
    return - ((p_pass/set)*np.log2(p_pass/set)) - ((p_fails/set)*np.log2(p_fails/set))

# this fuction calculates the best info gain of each attribute in the dataset      
def calculate_BestInfoGain(df): #Atribute is the column of the data frame
    
    label = "label"
    label_counts = df[label].value_counts()
    # print(label_counts)
    label_counts = label_counts.to_dict()
    
    label_pass = label_counts["e"]
    label_fail = label_counts["p"]
    #calculate the Information gain for each attribute
    infoGain_values = {}
    for (attribute, set)  in df.iteritems():
        
        if(attribute == label): continue
        features = df[attribute].unique()
        # calculating the number of pass and fail labels to compute entropy for each value of Attribute
        average = 0
        s = len(df)
        # keep track of attribute with highest gain
        
        for x in features:
            p_pass = df[(df[attribute] == x) & (df[label] == "e")]
            p_pass = len(p_pass)
            p_fails = df[(df[attribute] == x) & (df[label] == "p")]
            p_fails = len(p_fails)

            sv = p_pass + p_fails
            Hs = entropy(p_pass,p_fails,sv)
            average += Hs * (p_pass + p_fails) / s

        infoGain = entropy(label_pass,label_fail, s) - average 
        infoGain_values[attribute] = infoGain
    
    #getting the attribute with the highest gain
    best_attribute = max(infoGain_values, key=lambda k: infoGain_values[k])
    best_gain = infoGain_values[best_attribute]
    return best_attribute,best_gain


def ID3(S,attributes,max_length="",current_legth=1):
    label_column = 'label'
    labels= S[label_column]
    unique_values_labels = np.unique(labels)
    unique_len = len(unique_values_labels)
   # recursion base condition: all labels are the same therefore return node with label
    if(unique_len == 1 or current_legth == max_length): 
        return DecisionTreeNode(name="leaf",label=unique_values_labels[0])

    # else create note with atributes and split base on entropy 
    best_attribute,best_gain = calculate_BestInfoGain(S)
    if(current_legth == 1):
        print(f"root node: {best_attribute} \nInformation gain:{best_gain:.3f}")

    # get branch nodes
    nodes_bestBranch = S[best_attribute]
    nodes_bestBranch = np.unique(nodes_bestBranch)
    node_name = 'node'
    rootNode = DecisionTreeNode(name=node_name,informationGain=best_gain,branches={},feature=best_attribute)
    # for each value of the attribute in rootNode (Smell)   
    
    for node in nodes_bestBranch:
        # subset of values with the node there, or rows with that value in it
        sv = S[(S[best_attribute] == node)]
        #add common label in S to node when row doesn't exist
        if(sv.empty):  
            mostcommon_label = labels.mode()
            rootNode.branches[node] =  DecisionTreeNode(name='empty',label=mostcommon_label) 
        # drop A (best attribute from Set)
        remaining_attributes = sv.drop(columns=best_attribute)
        rootNode.branches[node] =  ID3(sv,remaining_attributes,max_length,current_legth + 1)# recursion 
        
    return rootNode

def export_df(df,filename):
    df.to_csv(filename + '.csv', sep='\t', index=False)


def predict(tree, data_point):
    # Traverse the tree until a leaf node is reached
    # Traverse the tree until a leaf node is reached
    while tree and tree.branches:
        feature_value = data_point[tree.feature]
        tree = tree.branches.get(feature_value, None)
    # Check if the current node is a leaf node
    if tree:
        return tree.label
    else:
        # Handle the case where tree is None (no prediction can be made)
        return None

def calculate_accuracy(predicted_labels,actual_labels):
    if len(predicted_labels) != len(actual_labels):
        raise ValueError("labels must be have same length")

    correct_count = sum(1 for a, b in zip(predicted_labels, actual_labels) if a == b)
    total_elements = len(actual_labels)

    accuracy = correct_count / total_elements
    return accuracy

def evaluate(dataset,model):
    label_column = 'label'
    predicted_labels = []
    #predic value using the model and add to array to further compare
    for row in range(len(dataset)):
        predicted_label = predict(model,dataset.iloc[row])
        predicted_labels.append(predicted_label)
    labels = dataset[label_column].to_numpy() #actual labels
    accuracy_training = calculate_accuracy(predicted_labels, labels)
    print(f"Accuracy: {accuracy_training:.3f} \n")
    return accuracy_training

def standard_deviation(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_deviation = variance ** 0.5
    return std_deviation

def kfold_crossval_tree(k_array,depth):
    x_validations = []
    for i, val in enumerate(k_array): #here we are iterating over each possible k-fold train and val
        print("K fold Iteration : ", i+1)
        training_set = k_array[:i] + k_array[i+1:] 
        validation_set = [val]
        
        # create combined df 
        k_train = pd.DataFrame(columns=k_array[0].columns)
        k_train = k_train.append(training_set, ignore_index=True)
        k_test = pd.DataFrame(columns=k_array[0].columns)
        k_test = k_test.append(validation_set, ignore_index=True)

        # build model with k_train
        label_column = 'label'
        attributes = k_train.drop(columns=label_column)
        k_model = ID3(k_train,attributes,depth)
        # evaluate k_model in validation set
        x_validation = evaluate(k_test,k_model)
        x_validations.append(x_validation)
    average_xvalidation = sum(x_validations) / len(x_validations)
    std_dev = standard_deviation(x_validations)
    print(f"standard_deviation x-validation: {std_dev:.3f}")
    print(f"average x-validation: {average_xvalidation:.3f}\n")
    return average_xvalidation

def main():
    
    #reading data
    training_set = pd.read_csv('data/train.csv')
    test_set = pd.read_csv('data/test.csv')

    #filling up missing attributes on column stalk-root 
    # with most common label in training and test set
    label1 = training_set["stalk-root"]
    mostcommon_label_training = label1.mode()
    label2 = training_set["stalk-root"]
    mostcommon_label_test = label2.mode()
    training_set["stalk-root"] = training_set["stalk-root"].replace('?', mostcommon_label_training[0])
    test_set["stalk-root"] = test_set["stalk-root"].replace('?', mostcommon_label_test[0])
    
    label_column = 'label'
    attributes = training_set.drop(columns=label_column)

    #Evaluating tree on data
    print("------------------------")
    print("**Full Tree**")
    print("------------------------")
    #calculate entropy of data
    p_values= test_set[label_column].value_counts()
    data_entropy = entropy(p_values[0],p_values[1],len(test_set))
    print(f"entropy: {data_entropy}")
    #building the model with a decision Tree
    decision_tree = ID3(training_set,attributes)
    print("--Training data--")
    evaluate(training_set,decision_tree)
    print("--Test data--")
    evaluate(test_set,decision_tree) 
    print(f"Tree depth: {decision_tree.depth()}")

    print("------------------------")
    print("**K fold cross validations**\n")
    #reading in kfold data and putting it in an array
    fold1 = pd.read_csv('data/CVfolds_new/fold1.csv')  
    fold2 = pd.read_csv('data/CVfolds_new/fold2.csv')    
    fold3 = pd.read_csv('data/CVfolds_new/fold3.csv')    
    fold4 = pd.read_csv('data/CVfolds_new/fold4.csv')    
    fold5 = pd.read_csv('data/CVfolds_new/fold5.csv')    
    k_array = [fold1,fold2,fold3,fold4,fold5]
    depths = [1, 2, 3, 4, 5, 10, 15]
    #cross validation with different depths
    for depth in depths:
        print(f"cross validation with depth {depth}")
        print("------------------------------")
        kfold_crossval_tree(k_array,depth=depth)
        print("------------------------------")

    print("best depth is 5")
    #using depth 5 to build tree on training set
    print()
    print("limiting set info on full data")
    print("------------------------------")
    decision_tree = ID3(training_set,attributes,max_length=4)
    print("------------------------------")
    print("Accuracy of trained classifier in training set")
    evaluate(training_set,decision_tree)
    print("------------------------------")
    print("Accuracy of trained classifier in test set")
    evaluate(test_set,decision_tree)
    print("------------------------------")

if __name__ == "__main__":
    
    main()
