import csv
from demos import cmd
from sklearn import svm,preprocessing,tree
import pandas
from pdb import set_trace
import numpy as np

def make_csv(file='MDP_Original_data2.csv'):
    with open(file, "r") as csvfile:
        content = [x for x in csv.reader(csvfile, delimiter=',')]
    fields = content[0]
    tmp=0
    for that in content[1:][::-1]:
        if that[4]=="WE":
            that[4]=1
        else:
            that[4]=0
        if float(that[5])!=0:

            if float(that[5])>0:
                that[5]=1
                tmp=1
            else:
                that[5]=-1
                tmp=-1
        else:
            that[5]=tmp
    with open("supervised_data.csv", "w") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for c in content:
            csvwriter.writerow(c)

def select_SVM(file='supervised_data.csv'):
    original_data = pandas.read_csv(file)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1
    features = feature_name[start_Fidx: len(feature_name)]+['priorTutorAction']
    mat=original_data[features].values
    min_max_scaler = preprocessing.MinMaxScaler()
    mat = min_max_scaler.fit_transform(mat)
    clf = svm.SVC(kernel='linear')
    clf.fit(mat,original_data['reward'])
    order = np.argsort(np.abs(clf.coef_[0]))[::-1][:8]
    new_dt = original_data[feature_name[:start_Fidx]+list(np.array(features)[order])]
    for key in np.array(features)[order]:
        if len(new_dt[key].unique())>6:
            thres = new_dt[key].median()
            for i,x in enumerate(new_dt[key]):
                if x>= thres:
                    new_dt.set_value(i,key,1)
                else:
                    new_dt.set_value(i,key,0)
    old_data = pandas.read_csv('MDP_Original_data2.csv')
    new_dt['reward']=old_data['reward']
    new_dt['priorTutorAction']=old_data['priorTutorAction']
    new_dt.to_csv('SVM_output.csv',index=False)

    return np.array(features)[order]

def select_DT(file='supervised_data.csv'):
    original_data = pandas.read_csv(file)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1
    clf = tree.DecisionTreeClassifier(max_depth=3)

    features = feature_name[start_Fidx: len(feature_name)]+['priorTutorAction']
    clf.fit(original_data[features],original_data['reward'])


    dot_data= tree.export_graphviz(clf,out_file=None,feature_names=features)
    print(dot_data)
    want={}

    for node in dot_data.split(';'):
        try:

            rest = node.split('label=\"')[1]
            value=float(rest.split(' <= ')[1].split('\\')[0])
            key=rest.split(' <= ')[0]
            want[key]=value
        except:
            continue
    new_dt = original_data[feature_name[:start_Fidx]+want.keys()]
    for key in want:
        thres= want[key]
        for i,x in enumerate(new_dt[key]):
            if x> thres:
                new_dt.set_value(i,key,1)
            else:
                new_dt.set_value(i,key,0)

    old_data = pandas.read_csv('MDP_Original_data2.csv')
    new_dt['reward']=old_data['reward']
    new_dt['priorTutorAction']=old_data['priorTutorAction']

    new_dt.to_csv('DT_output.csv',index=False)

    return want

def discretize(file='MDP_Original_data2.csv'):
    original_data = pandas.read_csv(file)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1
    features = feature_name[start_Fidx: len(feature_name)]+['priorTutorAction']
    for key in features:
        if len(original_data[key].unique())>6:
            thres = original_data[key].median()
            for i,x in enumerate(original_data[key]):
                if x> thres:
                    original_data.set_value(i,key,1)
                else:
                    original_data.set_value(i,key,0)

    original_data.to_csv('dis_data.csv',index=False)

def forward(file='dis_data.csv'):
    from MDP_function2 import induce_policy_MDP2
    original_data = pandas.read_csv(file)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1
    features = feature_name[start_Fidx: len(feature_name)]
    fea_num=8
    selected_feature=[]
    for fea_id in xrange(fea_num):
        best=-1
        best_fea=''
        for fea in features:
            try:
                ECR_value = induce_policy_MDP2(original_data, selected_feature+[fea])
            except:
                ECR_value = 0
            if ECR_value>best:
                best_fea=fea
                best=ECR_value
        print('%s: %f' %(best_fea,best))
        selected_feature.append(best_fea)
        features.remove(best_fea)
    new_dt = original_data[feature_name[:start_Fidx]+selected_feature]
    new_dt.to_csv('forward_output.csv',index=False)


def GA(file='dis_data.csv'):
    ## Genetic Algorithm
    from MDP_function2 import induce_policy_MDP2
    import random

    def eval(x):
        if not x in memory:
            try:
                memory[x]=induce_policy_MDP2(original_data, list(x))
            except:
                memory[x]=0
        return memory[x]

    original_data = pandas.read_csv(file)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1
    features = np.array(feature_name[start_Fidx: len(feature_name)])
    fea_num=8

    ### GA parameters
    NumG=100
    MaxG=20
    mutate=0.1
    memory={}


    ### Initial candidates
    current={}
    for i in xrange(NumG):
        x=tuple(set(np.random.choice(features, fea_num, replace=False)))
        current[x]=eval(x)
    lastE = len(memory)
    ### Generations
    for generation in xrange(MaxG):
        ## selection
        survivor=np.array(memory.keys())[np.argsort(memory.values())[::-1][:int(NumG/10)]]
        survivor=map(set,survivor)
        survivor=map(tuple,survivor)
        ## crossover
        parents=set([])
        for what in survivor:
            parents=parents|set(what)
        parents=tuple(parents)
        current={}
        for j in xrange(NumG):
            if random.random()<mutate:
                ## mutate
                new=tuple(set(np.random.choice(features,fea_num,replace=False)))
            else:
                new = tuple(set(np.random.choice(parents,fea_num,replace=False)))
            current[new]=eval(new)
        if lastE==len(memory):
            break
        lastE=len(memory)
    best=tuple(set(np.array(memory.keys())[np.argsort(memory.values())[::-1][0]]))
    print(best)
    print(memory[best])
    new_dt = original_data[feature_name[:start_Fidx]+list(best)]
    new_dt.to_csv('GA_output.csv',index=False)














if __name__ == "__main__":
    eval(cmd())