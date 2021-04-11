import numpy as np
import scipy
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import project_functions2 as pf

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_multiple_decision_trees(X, y, minimum_depth, maximum_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    columns = ['Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0

    for depth in range(minimum_depth, maximum_depth + 1):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        df.loc[depth] = [train_score, test_score]
        # if curr_score > max_score:
        #    max_score = curr_score
        #    max_score_depth = depth

    plt.figure()
    df.plot()
    plt.axis([1, maximum_depth, 0, 1.1])
    plt.ylabel('Score')
    plt.xlabel('Depth')
    plt.title('Training and Test Scores vs Depth')
    plt.show()

    return df


def decision_tree_post_pruning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()


def train_decision_trees_multiple_train_test(X, y, depth, min_X, max_X):
    columns = ['Training Size', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    training_list = []
    while min_X <= max_X:
        training_list.append(min_X)
        min_X += 0.05
    training_list.append(max_X)

    for curr_train in training_list:
        # print(curr_train)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=curr_train, random_state=42)
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        df.loc[len(df)] = [curr_train, train_score, test_score]
        # if curr_score > max_score:
        #    max_score = curr_score
        #    max_score_depth = depth

    plt.figure()
    df.plot(x='Training Size')
    # plt.axis([1,max_X,0,1.1])

    plt.ylabel('Score')
    plt.xlabel('Training Split')
    plt.title('Training and Test Scores vs Training Split')
    plt.show()

    return df



def train_multiple_random_forests(combine_df, split_time, stock_dfs, min_forest_size, max_forest_size, min_features, max_features, forest_step=10):
    X_train, y_train, X_test, y_test = pf.multi_stock_train_test_split(combine_df, split_time, stock_dfs)

    columns = ['Forest Size', 'Features', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    for num_features in range(min_features, max_features + 1):
        for forest_size in range(min_forest_size, max_forest_size + 1, forest_step):
            clf = RandomForestClassifier(n_estimators=forest_size, max_features=num_features,
                                         random_state=0).fit(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            train_score = clf.score(X_train, y_train)
            df.loc[len(df)] = [forest_size, num_features, train_score, test_score]

    return df


def train_random_forests_multiple_train_test(X, y, forest_size, features, min_X, max_X):
    columns = ['Training Size', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    training_list = []
    while min_X <= max_X:
        training_list.append(min_X)
        min_X += 0.05
    training_list.append(max_X)

    for curr_train in training_list:
        # print(curr_train)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=curr_train, random_state=42)
        clf = RandomForestClassifier(n_estimators=forest_size, max_features=features, \
                                     random_state=0).fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        train_score = clf.score(X_train, y_train)
        df.loc[len(df)] = [curr_train, train_score, test_score]
        # if curr_score > max_score:
        #    max_score = curr_score
        #    max_score_depth = depth

    plt.figure()
    df.plot(x='Training Size')
    # plt.axis([1,max_X,0,1.1])

    plt.ylabel('Score')
    plt.xlabel('Training Split')
    plt.title('Training and Test Scores vs Training Split')
    plt.show()

    return df


def train_multiple_neural_networks(X, y, min_layers, max_layers, min_layer_size, max_layer_size, layer_size_step=10,
                                   solver='adam'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    columns = ['Hidden Layer Size', 'Number of Hidden Layers', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    for num_layers in range(min_layers, max_layers + 1):
        for layer_size in range(min_layer_size, max_layer_size + 1, layer_size_step):
            layer_list = [layer_size] * num_layers
            ## Uncomment lines below to produce nns with growing hidden layers and replace layer_list with final_layer_list below
            # final_layer_list = []
            # for i, value in enumerate(layer_list):
            #    curr_sum = value*(i+1)
            #    final_layer_list.append(curr_sum)
            # print(layer_list)
            nnclf = MLPClassifier(hidden_layer_sizes=layer_list,
                                  solver=solver, random_state=69).fit(X_train, y_train)
            test_score = nnclf.score(X_test, y_test)
            train_score = nnclf.score(X_train, y_train)
            df.loc[len(df)] = [layer_size, num_layers, train_score, test_score]

    return df


def train_neural_networks_multiple_train_test(X, y, layer_size, layers, solver, min_X, max_X):
    columns = ['Training Size', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    layer_list = [layer_size] * layers
    training_list = []
    while min_X <= max_X:
        training_list.append(min_X)
        min_X += 0.05
    training_list.append(max_X)

    for curr_train in training_list:
        # print(curr_train)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=curr_train, random_state=42)
        nnclf = MLPClassifier(hidden_layer_sizes=layer_list,
                              solver=solver, random_state=69).fit(X_train, y_train)
        test_score = nnclf.score(X_test, y_test)
        train_score = nnclf.score(X_train, y_train)
        df.loc[len(df)] = [curr_train, train_score, test_score]
        # if curr_score > max_score:
        #    max_score = curr_score
        #    max_score_depth = depth

    plt.figure()
    df.plot(x='Training Size')
    # plt.axis([1,max_X,0,1.1])

    plt.ylabel('Score')
    plt.xlabel('Training Split')
    plt.title('Training and Test Scores vs Training Split')
    plt.show()

    return df


def train_multiple_neural_networks_scaler(combine_df, split_time, stock_dfs, min_layers, max_layers, min_layer_size,
                                          max_layer_size, layer_size_step=10, solver='adam'):
    scaler = MinMaxScaler()
    activations = ['identity', 'logistic', 'tanh', 'relu']
    X = combine_df.iloc[:, :-1]
    y = combine_df.iloc[:, -1:]
    # X = stock_df['Days From IPO'].values.reshape(-1, 1)
    # y = stock_df['Close'].values.reshape(-1, 1)

    # Does train/Test Split on last year
    # Change the -50 to a differnt value to change split point
    split_mark = int(len(combine_df) - (split_time * len(stock_dfs)))
    X_train = X.head(split_mark)
    X_test = X.tail(len(combine_df) - split_mark)
    y_train = y.head(split_mark)
    y_test = y.tail(len(combine_df) - split_mark)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)




    columns = ['Hidden Layer Size', 'Number of Hidden Layers', 'activation', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    for act in activations:
        for num_layers in range(min_layers, max_layers + 1):
            for layer_size in range(min_layer_size, max_layer_size + 1, layer_size_step):
                layer_list = [layer_size] * num_layers
                # print(layer_list)
                nnclf = MLPRegressor(hidden_layer_sizes=layer_list, activation=act,
                                      solver=solver, random_state=69).fit(X_train_scaled, y_train)
                test_pred = nnclf.predict(X_test_scaled)
                train_pred = nnclf.predict(X_train_scaled)

                test_score = r2_score(y_test.dropna(), test_pred[:len(y_test.dropna())])
                train_score = r2_score(y_train, train_pred)

                df.loc[len(df)] = [layer_size, num_layers, act, train_score, test_score]

    return df


def train_scaled_neural_networks_multiple_train_test(X, y, layer_size, layers, solver, min_X, max_X):
    scaler = MinMaxScaler()
    columns = ['Training Size', 'Training Score', 'Test Score']
    df = pd.DataFrame(columns=columns)
    max_score = 0
    max_score_depth = 0
    layer_list = [layer_size] * layers
    training_list = []
    while min_X <= max_X:
        training_list.append(min_X)
        min_X += 0.05
    training_list.append(max_X)

    for curr_train in training_list:
        # print(curr_train)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=curr_train, random_state=42)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nnclf = MLPClassifier(hidden_layer_sizes=layer_list,
                              solver=solver, random_state=69).fit(X_train, y_train)
        test_score = nnclf.score(X_test, y_test)
        train_score = nnclf.score(X_train, y_train)
        df.loc[len(df)] = [curr_train, train_score, test_score]
        # if curr_score > max_score:
        #    max_score = curr_score
        #    max_score_depth = depth

    plt.figure()
    df.plot(x='Training Size')
    # plt.axis([1,max_X,0,1.1])

    plt.ylabel('Score')
    plt.xlabel('Training Split')
    plt.title('Training and Test Scores vs Training Split')
    plt.show()

    return df


def three_d_plot(x, y, z, title='Graph Title', x_name='X_axis', y_name='Y_axis', z_name='Z_axis'):
    x1 = np.linspace(x.min(), x.max(), len(x.unique()))
    y1 = np.linspace(y.min(), y.max(), len(y.unique()))

    x2, y2 = np.meshgrid(x1, y1)

    # Interpolate unstructured D-dimensional data.
    z1 = scipy.interpolate.griddata((x, y), z, (x2, y2), method='cubic')

    # Ready to plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x2, y2, z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    plt.show()