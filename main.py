import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, interpolate
from itertools import cycle
import math
import scipy.cluster.hierarchy as shc
from six import StringIO
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus


df = pd.read_csv("hotels_data.csv")
# le = LabelEncoder()
enc = OrdinalEncoder()


def main():
    # Task_1
    # add_features()
    df = pd.read_csv("hotels_data_Changed.csv")

    # Task_2
    # X_train,X_test,y_train,y_test = classification_train_test()
    # decision_tree(X_train,y_train,X_test,y_test)
    # naive_bayes(X_train,y_train,X_test,y_test)

    # Task_3

    # create_clustering_data()
    hierarchical_clustering()


#Task 1
def add_features():
    a = pd.to_datetime(pd.Series(df['Snapshot Date']), format='%m/%d/%Y 0:00')
    b = pd.to_datetime(pd.Series(df['Checkin Date']), format='%m/%d/%Y 0:00')
    df.drop('Snapshot Date', axis = 1, inplace = True)
    df.drop('Checkin Date', axis = 1, inplace = True)
    df['Snapshot Date'] = a
    df['Checkin Date'] = b
    DayDiff =[]
    WeekDay =[]
    DiscountDiff =[]
    DiscountPerc  =[]

    for i, row in df.iterrows():
        DayDiff.append((row['Checkin Date']-row['Snapshot Date']).days)
        WeekDay.append(row['Checkin Date'].strftime("%A"))
        DiscountDiff.append(row['Original Price']-row['Discount Price'])
        DiscountPerc.append((1-(row['Discount Price']/row['Original Price']))*100)

    df['DayDiff'] = DayDiff
    df['WeekDay'] = WeekDay
    df['DiscountDiff'] = DiscountDiff
    df['DiscountPerc'] = DiscountPerc
    df.to_csv('Hotels_data_Changed.csv')


#Task 2
def classification_train_test():
    df2 = pd.read_csv("hotels_data_Changed.csv",
                      usecols=['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name','DiscountPerc',
                               'Discount Code'])
    df2.groupby(['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name'])
    df2 = df2.sort_values(by='DiscountPerc',ascending=False,na_position='first')
    df2 = df2.drop_duplicates(subset=['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name'])
    X = df2[['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name']]
    # X = X.apply(lambda col: le.fit_transform(col))
    enc.fit(X)
    X = enc.transform(X)
    print(X)
    y = df2['Discount Code']
    y = pd.get_dummies(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)  # 70% training and 30% te
    return X_train, X_test, y_train, y_test

def decision_tree(X_train, y_train,X_test, y_test):
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # plot_statistics(y_test,y_pred)
    print("Accuracy decision tree:",metrics.accuracy_score(y_test,y_pred))

    # dot_data = StringIO()
    # export_graphviz(clf,out_file=dot_data,
    #                 filled=True,rounded=True,
    #                 special_characters=True,feature_names=['WeekDay','Snapshot Date','Checkin Date','DayDiff','Hotel Name'],class_names=['1','2','3','4'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('diabetes.png')



def naive_bayes(X_train, y_train,X_test, y_test):
    gnb = GaussianNB()
    print(X_train.shape)
    y_train = y_train.idxmax(axis=1)
    y_test = y_test.idxmax(axis=1)
    print(y_train.shape)
    print(y_train)
    y_pred = gnb.fit(X_train,y_train).predict(X_test)
    # print("Number of mislabeled points out of a total %d points : %d",(X_test.shape[0],(y_test != y_pred).sum()))
    print("Accuracy naive bayes:",metrics.accuracy_score(y_test,y_pred))
    # roc = metrics.roc_auc_score(y_test,y_pred)
    # plot_statistics(y_test,y_pred)

def plot_statistics(y_test,y_pred,n_classes = 4):
    y_test = y_test.to_numpy()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i],tpr[i],_ = metrics.roc_curve(y_test[:,i],y_pred[:,i])
        roc_auc[i] = metrics.auc(fpr[i],tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"],tpr["micro"],_ = metrics.roc_curve(y_test.ravel(),y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"],tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp1d(all_fpr,fpr[i],tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"],tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"],tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink',linestyle=':',linewidth=4)

    plt.plot(fpr["macro"],tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy',linestyle=':',linewidth=4)
    lw = 2

    colors = cycle(['aqua','darkorange','cornflowerblue'])
    for i,color in zip(range(n_classes),colors):
        plt.plot(fpr[i],tpr[i],color=color,lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i,roc_auc[i]))

    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()







    # # Compute ROC curve and ROC area for each class
    # y_test = y_test.to_numpy()
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:,i],y_pred[:,i])
    #     roc_auc[i] = metrics.auc(fpr[i],tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(),y_pred.ravel())
    # roc_auc["micro"] = metrics.auc(fpr["micro"],tpr["micro"])
    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr["micro"],tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]))
    # for i in range(n_classes):
    #     plt.plot(fpr[i],tpr[i],label='ROC curve of class {0} (area = {1:0.2f})'
    #                                  ''.format(i,roc_auc[i]))
    #
    # plt.plot([0,1],[0,1],'k--')
    # plt.xlim([0.0,1.0])
    # plt.ylim([0.0,1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

def create_clustering_data():
    hotels = df["Hotel Name"].value_counts().head(150).index.tolist()
    temp_df = df[df["Hotel Name"].isin(hotels)]
    # print(new_df)
    dates = temp_df["Checkin Date"].value_counts().head(40).index.tolist()
    temp_df = temp_df[temp_df["Checkin Date"].isin(dates)]
    # print(temp_df)
    columns = ["Hotel Name"]
    date_cols = [i for i in range(1,161)]
    columns.extend(date_cols)
    # dates.pop(0)
    new_df = pd.DataFrame(columns=columns)
    for h in hotels:
        row = []
        for d in dates:
            for n in [1,2,3,4]: #the 4 discount codes
                # price = temp_df[(df["Hotel Name"] == h)]
                # price = price[df["Checkin Date"] == d]
                # price = price[df["Discount Code"] == n]
                price = (temp_df[(temp_df["Hotel Name"] == h) & (temp_df["Checkin Date"] == d) & (temp_df["Discount Code"] == n)]['Discount Price'].min())
                if math.isnan(price):
                    price = -1
                row.append(price)
        input_row = [((i - min(i for i in row if i >= 0))/(max(row)-min(row)))*100 if i != -1 else i for i in row]
        # row = [(float(i) / max(row))*100 if i != -1 else i for i in row]
        input_row.insert(0,h)
        a_series = pd.Series(input_row,index=new_df.columns)
        new_df = new_df.append(a_series,ignore_index=True)
        # new_df = new_df.append(row)
        # print("done")

    print(new_df.head(5))
    new_df.to_csv('Clustering_data.csv')

def hierarchical_clustering():
    data_scaled = pd.read_csv('Clustering_data.csv',)
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograms")
    plt.xlabel('sample index')
    only_data = data_scaled.set_index("Hotel Name")
    only_data = only_data.drop("Unnamed: 0", axis=1)
    res = shc.linkage(only_data, method='ward')
    names = data_scaled["Hotel Name"].tolist()
    dend = shc.dendrogram(res, no_plot=True)
    lables_names = [names[i] for i in dend["leaves"]]
    lables_names = np.asarray(lables_names)
    # shc.dendrogram(res, labels=lables_names)
    # plt.savefig('Clustering_Dendrogram.png')
    # plt.show()
    fancy_dendrogram(
        res,
        truncate_mode='lastp',
        p=9,
        leaf_rotation=90.,
        leaf_font_size=8.,
        show_contracted=True,
        annotate_above=10,
        labels=lables_names,
        max_d=500
        # useful in small plots so annotations don't overlap
    )
    # shc.dendrogram(res,labels=lables_names,truncate_mode='lastp',p=7,leaf_rotation=90.,leaf_font_size=8.,
    # show_contracted=True,)  # to get a distribution impression in truncated branches)
    plt.savefig('Clustering_Dendrogram3.png')
    plt.show()




def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = shc.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Hotel name or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


if __name__ == "__main__":
    main()
