# presents: ofri rom:208891804,Avigail shekasta:209104314,Dan monsonego:313577595
# import libs
import pickle
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# import and install function automatically import and install the require libs
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', package])
        else:
            pip._internal.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


# Imports
import numpy as np
import pandas as pd

install_and_import('pandas')
install_and_import('numpy')
install_and_import('sklearn')
install_and_import('info_gain')
install_and_import('pyitlib')

# import of models from sklearn
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# define dictionaries we use in this project d contain the dataframe
d = {}
model_dict = {}
result_dict = {}
Class_name = {}


# get_df function is get the path to the csv file and create dataframe object
def get_df(path):
    # function return the data frame from the relevant path
    d["df"] = pd.read_csv(path)


# drop rows function drop all the rows that contain NaN value in the class column
def drop_rows(class_name):
    d["df"] = d["df"][d["df"][class_name].notna()]
    d["df"] = d["df"].reset_index(drop=True)
    Class_name['class'] = class_name


# fill mean value
def fill_mean_values(data, class_name):
    data[class_name] = data[class_name].fillna(data[class_name].mean()[0])


# fill mode value
def fill_mode_values(data, class_name):
    data[class_name] = data[class_name].fillna(data[class_name].mode()[0])


# this function called from main fill data function and call to fill functions
def sub_fill_data(data):
    for i in d['df'].columns:
        if d["df"][i].dtype is type(int) and d["df"][i].nunique() > 10:
            fill_mean_values(data, i)
        else:
            fill_mode_values(data, i)


# by the choice of the user this function perform the filling by calling the sub data fill function(related to all
# the data or related to the class column values)
def main_fill_data(choice, class_name):
    result_dict['data cleaning'] = 'data cleaning'
    if choice == 1:
        sub_fill_data(d["df"])
    else:
        c = 0
        d["df"] = d["df"].sort_values(by=class_name)
        d['df'] = d['df'].reset_index(drop=True)
        temp = d["df"].iloc[0, 0]
        print(d["df"].iloc[0, 0] + "\n")
        for j in range((d["df"].shape[0]) + 1):
            if temp != d["df"].iloc[j, 0]:
                d["kp"] = d["df"][0:j - 1]
                sub_fill_data(d["kp"])
                d["df"][0:j - 1] = d["kp"]

                d["kp"] = d["df"][j:(d["df"].shape[0])]
                sub_fill_data(d["kp"])
                d["df"][j:(d["df"].shape[0])] = d["kp"]
                return


# Conversion_to_number function take the dataframe object and using the label encoder to convert the data into numeric type
def Conversion_to_number():
    le = preprocessing.LabelEncoder()
    for i in d["df"].columns:
        d["df"][i] = le.fit_transform(d["df"][i])


# Normalization function take col name as argument and perform min max normalization to the specific column
def Normalization(class_name):
    d["df"][class_name] = preprocessing.minmax_scale(d["df"][class_name])
    result_dict['Normalization'] = 'Normalization: ' + class_name


# Equal_width function
def Equal_width(col, k):
    d["df"] = d["df"].sort_values(by=col)
    rLst = [x for x in range(min(d["df"][col]), max(d["df"][col]) + 1, (max(d["df"][col]) - min(d["df"][col])) // k)]
    result = []
    for i in d["df"][col]:
        found = False
        for j in range(len(rLst) - 1):
            if i in [x for x in range(rLst[j], rLst[j + 1] + 2)]:
                result += [str([rLst[j], rLst[j + 1]])]
                found = True
                break
        if not found:
            result += [str([i - 1, i])]
    d["df"][col] = result
    result_dict['Equal_width'] = 'Equal_width: ' + col


# Equal_frequency_discretization
def Equal_frequency_discretization(col, k):
    if k < len(d["df"][col]):
        d["df"][col] = sorted(d["df"][col])
        r = len(d["df"][col]) // (k)
        l = len(d["df"][col]) % k
        result = [d["df"][col][x:x + r - 1] for x in range(0, len(d["df"][col]), r)]
        for i in range(0, len(d["df"][col]), r):
            j = i
            while (j != i + r):
                d["df"][col][j] = i
                j = j + 1
        i = len(d["df"][col]) - l
        j = i
        while (j != len(d["df"][col])):
            d["df"][col][j] = i
            j = j + 1
    result_dict['Equal_frequency_discretization'] = 'Equal_frequency_discretization: ' + col


# entrophy_based_binning
def entrophy_based_binning(col, k):
    import entropy_based_binning as ebb
    d["df"][col] = ebb.bin_array(d["df"][col], nbins=k, axis=0)
    result_dict['entrophy_based_binning'] = 'entrophy_based_binning: ' + col


# this function split the data into train and test parts when the train part is 20% of the data
# the function return the split results in dictionaries
def split_data(data, class_name):
    X = data.drop([class_name], axis=1)
    y = data[class_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# ------------------ our id3 implement model
# -------------------------------------------------------------------------------------
# recursive function that create the tree by calculate the info gain every each time and choose the best features
def id3_by_us(data, original_data, features, class_name, parent_node_class=None):
    if len(np.unique(data[class_name])) <= 1:
        return np.unique(data[class_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[class_name])[
            np.argmax(np.unique(original_data[class_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[class_name])[np.argmax(np.unique(data[class_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, class_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3_by_us(sub_data, d['df'], features, class_name, parent_node_class)
            tree[best_feature][value] = subtree
        return (tree)


# this function create the result prediction of the tree model
def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


# this function test the result of the tree model and return the positive and negative results
def test(data, tree):
    target = Class_name['class']
    queries = data.iloc[:, 1:].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    test_data = pd.DataFrame(columns=['class'])
    # values to print the confusion matrix
    predict_list = []
    test = []
    data = data[target].reset_index(drop=True)
    print(data)
    for i in range(len(data)):
        if data[i] == 1:
            test.append(1)
        else:
            test.append(0)
        test_data.loc[i, "class"] = data[i]
        if int(predict(queries[i], tree, 1.0)) == 1:
            predict_list.append(1)
        else:
            predict_list.append(0)
        predicted.loc[i, "predicted"] = int(predict(queries[i], tree, 1.0))
    positive = np.sum(predicted["predicted"] == test_data["class"])
    negative = np.sum(predicted["predicted"] != test_data["class"])
    model_dict['predict'] = predict_list
    model_dict['y_test'] = test
    model_dict['model'] = tree
    model_dict['type'] = 'our id3'
    print(positive, negative)
    return positive, negative


# calculate the entropy of column
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# this function use the entropy function to calculate the info gain for the column
def InfoGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# ------------------------ our id3 implement model
# -------------------------------------------------------------------------------------

# id3 model from sklearn lib
def id3(class_name):
    train_test = split_data(d["df"], class_name)
    X_train = train_test['X_train']
    y_train = train_test['y_train']
    y_test = train_test['y_test']
    X_test = train_test['X_test']
    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    model_dict["predict"] = predict
    model_dict["model"] = model
    model_dict["y_test"] = y_test
    model_dict["type"] = "id3"
    model_dict["X_train"] = X_train
    model_dict["y_train"] = y_train


# dictionaries for our naive bayes model
no_dict = {}
yes_dict = {}


# our implement for naive bayes model
def Naive_bayes_by_us(class_name):
    df = d['df']
    X = df.drop([class_name], axis=1)
    y = df[class_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    results = y_train.value_counts()
    df = X_train
    yes = results[0]
    no = results[1]
    x = d["df"][class_name].unique()

    y = x[0]
    x = x[1]

    total = yes + no
    for i in df.columns:
        lst = df[i].unique()
        for j in lst:
            sum_n = 0
            sum_y = 0
            for w in range(total):
                if df[i][w] == j and y_train[w] == x:
                    sum_n += 1

                if df[i][w] == j and y_train[w] == y:
                    sum_y += 1

            sum_n /= no
            no_dict[i + str(j)] = sum_n
            sum_y /= yes
            yes_dict[i + str(j)] = sum_y

        no_dict[x] = no / total
        yes_dict[y] = yes / total
    for i in d["df"].columns:
        for j in d["df"][i].unique():
            p = str(i) + str(j)
            if p not in no_dict:
                no_dict[p] = 1
            if p not in yes_dict:
                yes_dict[p] = 1

    c = 0
    cc = []
    for i, rows in X_test.iterrows():
        pp = ''.join([i for i in str(i) if not i.isdigit()])
        p = pp + str(rows)
        p = (p.replace(" ", "")).replace("\n", " ").split()
        p = p[:-1]

        R = query(p, class_name)
        y_test = y_test.reset_index(drop=True)
        if R == y:
            cc.append(y)
        else:
            cc.append(x)
        c = c + 1
    le = preprocessing.LabelEncoder()
    y_test = le.fit_transform(y_test)
    model_dict['predict'] = cc
    model_dict["no_dict"] = no_dict
    model_dict["yes_dict"] = yes_dict
    model_dict["y_test"] = y_test
    model_dict["type"] = "Naive_bayes_by_us"


# this is the test function of our naive bayes model this function use in our model and make the prediction result
def query(list, class_name):
    x = d["df"][class_name].unique()
    y = x[0]
    x = x[1]
    pn = 1
    py = 1
    for i in list:
        py *= yes_dict[i]
        pn *= no_dict[i]
    py *= yes_dict[y]
    pn *= no_dict[x]
    if (py > pn):
        return y
    else:
        return x


# Naive bayes from sklearn
def Naive_bayes(class_name):
    train_test = split_data(d["df"], class_name)
    X_train = train_test['X_train']
    y_train = train_test['y_train']
    y_test = train_test['y_test']
    X_test = train_test['X_test']
    model = GaussianNB()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    model_dict["predict"] = predict
    model_dict["model"] = model
    model_dict["y_test"] = y_test
    model_dict["type"] = "Naive_bayes"
    model_dict["X_train"] = X_train
    model_dict["y_train"] = y_train


# knn from sklearn
def Knn(class_name, n_neighbors=5):
    train_test = split_data(d["df"], class_name)
    X_train = train_test['X_train']
    y_train = train_test['y_train']
    y_test = train_test['y_test']
    X_test = train_test['X_test']
    model = KNeighborsClassifier(n_neighbors)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    acc = accuracy_score(y_test, predict)
    model_dict["predict"] = predict
    model_dict["model"] = model
    model_dict["X_train"] = X_train
    model_dict["y_train"] = y_train
    model_dict["y_test"] = y_test
    model_dict["type"] = "Knn"


# k-means from sklearn
def kmeans(class_name, n_clusters=10):
    le = preprocessing.LabelEncoder()
    for i in d['df'].columns:
        d['df'][i] = le.fit_transform(d['df'][i])
    df = d['df'].to_numpy()
    pca = PCA(2)
    # Transform the data
    df = pca.fit_transform(df)
    print(df)
    kmeans = KMeans(n_clusters=n_clusters)
    # predict the labels of clusters.
    label = kmeans.fit_predict(df)
    u_labels = np.unique(label)
    # plotting the results:
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.legend()
    plt.savefig('fig.png')


# this function save the dataframe after the clean and fill actions into new csv file to the projects folder
def save():
    d["df"].to_csv('Clean_Data.csv', index=None, header=True)


# this function save the model data into a binary file
def pickl_model_save():
    filename = model_dict["type"]
    outfile = open(filename, 'wb')
    pickle.dump(model_dict, outfile)
    outfile.close()


# this function save the model result into a binary file
def pickl_matrix_etc():
    filename = "result_dict"
    outfile = open(filename, 'wb')
    pickle.dump(result_dict, outfile)
    outfile.close()


# this function create the performance matrix for the current model use the value from the dictionaries
def matrix_performace():
    cf_matrix = confusion_matrix(model_dict["y_test"], model_dict["predict"])
    result_dict['performance matrix'] = cf_matrix
    return cf_matrix


# this function create the train matrix for the current model use the value from the dictionaries
def matrix_train():
    target = Class_name['class']
    X = d['df']
    y = d['df'][target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cf_matrix = confusion_matrix(X_train[target], y_train)
    result_dict['train matrix'] = cf_matrix
    return cf_matrix


# this function create the test matrix for the current model use the value from the dictionaries
def matrix_test():
    target = Class_name['class']
    X = d['df']
    y = d['df'][target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cf_matrix = confusion_matrix(X_test[target], y_test)
    result_dict['test matrix'] = cf_matrix
    return cf_matrix


# this function calculate the accuracy score of each model
def acc():
    x = accuracy_score(model_dict["y_test"], model_dict["predict"])
    result_dict['accuracy test '] = x
    return x


def precision():
    x = precision_score(model_dict["y_test"], model_dict["predict"])
    result_dict['precision score '] = x
    return x


def recall():
    x = recall_score(model_dict["y_test"], model_dict["predict"])
    result_dict['recall score '] = x
    return x


def fmeasure():
    x = f1_score(model_dict["y_test"], model_dict["predict"])
    result_dict['f measure score '] = x
    return x


# this function calculate the majority test of our data and print the accuracy score according to the majority result
def majority_test(class_name):
    results = d['df'][class_name].value_counts()
    y_value = results[0]
    x_value = results[1]
    x = d["df"][class_name].unique()
    y = x[0]
    x = x[1]
    if y_value > x_value:
        class_value = y
    else:
        class_value = x
    predict = [class_value for x in range(len(model_dict["y_test"]))]
    acc = accuracy_score(model_dict["y_test"], predict)
    result_dict['majority test'] = acc
    print(acc)
