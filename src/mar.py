from __future__ import print_function, division
import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os

class MAR(object):
    def __init__(self):
        self.fea_num = 4000
        self.step = 10
        self.enough = 30
        self.kept=50
        self.atleast=100


    def create(self,filename):
        self.filename=filename
        self.name=self.filename.split(".")[0]
        self.flag=True
        self.hasLabel=True
        self.record={"x":[],"pos":[]}
        self.body={}
        self.header=[]
        self.est_num=[]
        self.lastprob=0
        self.offset=0.5
        self.interval=3
        self.last_pos=0
        self.last_neg=0

        self.loadfile()
        return self

    ### Use previous knowledge, labeled only
    def create_old(self, filename):
        with open("../workspace/coded/" + str(filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]
        for c in content[1:]:
            self.body["body"].append(c[:-2])
            self.body["time"].append(c[-1])
            self.body["code"].append(c[-2])

        self.csr_mat=np.array(self.body["body"])[:,3:-1].astype(float)
        self.body["label"]=['no' if x[-1]=='0' else 'yes' for x in self.body["body"]]
        self.last_pos=Counter(self.body["code"])["yes"]
        self.last_neg = Counter(self.body["code"])["no"]


    def loadfile(self):
        with open("../workspace/data/" + str(self.filename), "r") as csvfile:
            content = [x for x in csv.reader(csvfile, delimiter=',')]

        self.header = content[0]
        loc_i = self.header.index('loc')
        self.body["body"]=content[1:]
        self.body["loc"] = [int(x[loc_i]) for x in self.body["body"] if int(x[loc_i])>0]
        self.body["time"] = [0]*len(self.body["loc"])
        self.body["code"]=[-1]*len(self.body["loc"])
        tmp = range(3,loc_i)+range(loc_i+1,len(self.header)-1)
        self.csr_mat=np.array([np.array(x)[tmp].astype(float)/int(x[loc_i]) for x in self.body["body"] if int(x[loc_i])>0])

        self.body["label"]=[float(x[-1])/int(x[loc_i]) for x in self.body["body"] if int(x[loc_i])>0]

        return

    def get_numbers(self):
        self.pool = np.where(np.array(self.body['code']) == -1)[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        total = sum(self.body['loc'])
        found = sum([self.body['code'][x]*self.body['loc'][x] for x in self.labeled])
        cost = sum(np.array(self.body['loc'])[self.labeled])
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if cost>tmp:
            self.record['x'].append(cost)
            self.record['pos'].append(found)
        return found, cost, total

    def export(self):
        fields = ["Document Title", "Abstract", "Year", "PDF Link", "label", "code","time"]
        with open("../workspace/coded/" + str(self.name) + ".csv", "wb") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(fields)
            ## sort before export
            time_order = np.argsort(self.body["time"])[::-1]
            yes = [c for c in time_order if self.body["code"][c]=="yes"]
            no = [c for c in time_order if self.body["code"][c] == "no"]
            und = [c for c in time_order if self.body["code"][c] == "undetermined"]
            ##
            for ind in yes+no+und:
                csvwriter.writerow([self.body[field][ind] for field in fields])
        return

    ## save model ##
    def save(self):
        with open("memory/"+str(self.name)+".pickle","w") as handle:
            pickle.dump(self,handle)

    ## load model ##
    def load(self):
        with open("memory/" + str(self.name) + ".pickle", "r") as handle:
            tmp = pickle.load(handle)
        return tmp

    def estimate_curve(self,clf):
        ## estimate ##
        # self.est_num=Counter(clf.predict(self.csr_mat[self.pool]))["yes"]
        pos_at = list(clf.classes_).index("yes")
        prob = clf.predict_proba(self.csr_mat[self.pool])[:, pos_at]
        order = np.argsort(prob)[::-1]
        tmp = [x for x in np.array(prob)[order] if x > self.offset]

        ind = 0
        sum_tmp = 0
        self.est_num = []
        while True:
            tmp_x = tmp[ind * self.step:(ind + 1) * self.step]
            if len(tmp_x) == 0:
                break
            sum_tmp = sum_tmp + sum(tmp_x) - self.offset * len(tmp_x)
            self.est_num.append(sum_tmp)
            ind = ind + 1
            ##############
        try:
            past = np.argsort(self.body["time"])[::-1][:self.interval * self.step]
            self.lastprob = np.mean(clf.predict_proba(self.csr_mat[past])[:,pos_at])
            # self.lastprob = np.mean(np.array(prob)[order][:self.step])
        except:
            pass

    ## Train model ##
    def train(self,pne=False,cl="SVM-linear"):


        if cl.split('-')[0]=='SVM':
            clf = svm.SVC(kernel=cl.split('-')[1], probability=True)
        elif cl=="linear":
            from sklearn.linear_model import LinearRegression
            clf = LinearRegression()
        elif cl=="RF":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier()
        elif cl=="LR":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        elif cl=="NB":
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
        else:
            return [],[],self.random(),[]



        labels = np.array(self.body['label'])
        clf.fit(self.csr_mat[self.labeled], labels[self.labeled])


        certain_id, certain_prob = self.certain(clf)
        return certain_id, certain_prob

    ## reuse
    def train_reuse(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []

        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        # try:
        #     unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),self.atleast)),replace=False)
        # except:
        #     pass

        # print("%d,%d,%d" %(len(left),len(negs),len(unlabeled)))

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## not in use currently
    def train_reuse_random(self):
        thres=50

        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        pos, neg, total = self.get_numbers()
        if pos == 0 or pos + neg < thres:
            left=poses
            decayed = list(left) + list(negs)
        else:
            left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[self.last_pos:]]
            negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]
            decayed = list(left)+list(negs)
        clf.fit(self.csr_mat[decayed], np.array(self.body['code'])[decayed])
        ## aggressive undersampling ##
        if len(poses)>=self.enough:

            train_dist = clf.decision_function(self.csr_mat[negs])
            negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(left)]
            sample = list(left) + list(negs[negs_sel])
            clf.fit(self.csr_mat[sample], np.array(self.body['code'])[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        if pos == 0 or pos + neg < thres:
            return uncertain_id, uncertain_prob, np.random.choice(list(set(certain_id) | set(self.random())),
                                                                  size=np.min((self.step, len(
                                                                      set(certain_id) | set(self.random())))),
                                                                  replace=False), certain_prob
        else:
            return uncertain_id, uncertain_prob, certain_id, certain_prob

    def loc_sort(self):
        return self.pool[np.argsort(np.array(self.body['loc'])[self.pool])[:self.step]]

    ## Train_kept model ##
    def train_kept(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        ## only use latest poses
        left = np.array(poses)[np.argsort(np.array(self.body['time'])[poses])[::-1][:self.kept]]
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[poses])[::-1][:self.kept]]
        decayed = list(left)+list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed),self.atleast)), replace=False)
        except:
            pass

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## not in use currently
    def train_pos(self):
        clf = svm.SVC(kernel='linear', probability=True)
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]

        left = poses
        negs = np.array(negs)[np.argsort(np.array(self.body['time'])[negs])[self.last_neg:]]

        if len(left)==0:
            return [], [], self.random(), []

        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled,size=np.max((len(decayed),self.atleast)),replace=False)
        except:
            pass

        # print("%d,%d,%d" %(len(left),len(negs),len(unlabeled)))

        labels = np.array([x if x != 'undetermined' else 'no' for x in self.body['code']])
        all_neg = list(negs) + list(unlabeled)
        all = list(decayed) + list(unlabeled)

        clf.fit(self.csr_mat[all], labels[all])
        ## aggressive undersampling ##
        if len(poses) >= self.enough:
            train_dist = clf.decision_function(self.csr_mat[all_neg])
            pos_at = list(clf.classes_).index("yes")
            if pos_at:
                train_dist=-train_dist
            negs_sel = np.argsort(train_dist)[::-1][:len(left)]
            sample = list(left) + list(np.array(all_neg)[negs_sel])
            clf.fit(self.csr_mat[sample], labels[sample])
            self.estimate_curve(clf)

        uncertain_id, uncertain_prob = self.uncertain(clf)
        certain_id, certain_prob = self.certain(clf)
        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self,clf):
        prob = clf.predict(self.csr_mat[self.pool])
        order = np.argsort(prob)[::-1][:self.step]
        return np.array(self.pool)[order],np.array(prob)[order]


    ## Get random ##
    def random(self):
        return np.random.choice(self.pool,size=np.min((self.step,len(self.pool))),replace=False)

    ## Format ##
    def format(self,id,prob=[]):
        result=[]
        for ind,i in enumerate(id):
            tmp = {key: self.body[key][i] for key in self.body}
            tmp["id"]=str(i)
            if prob!=[]:
                tmp["prob"]=prob[ind]
            result.append(tmp)
        return result

    ## Code candidate studies ##
    def code(self,id,label):
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

    ## Plot ##
    def plot(self):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)

        fig = plt.figure()
        plt.plot(self.record['x'], self.record["pos"])

        plt.ylabel("Relevant Found")
        plt.xlabel("Documents Reviewed")
        name=self.name+ "_" + str(int(time.time()))+".png"

        dir = "./static/image"
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))

        plt.savefig("./static/image/" + name)
        plt.close(fig)
        return name

    def get_allbugs(self):
        return sum([self.body['label'][x]*self.body['loc'][x] for x in xrange(len(self.body['loc']))])

    ## Restart ##
    def restart(self):
        os.remove("./memory/"+self.name+".pickle")

