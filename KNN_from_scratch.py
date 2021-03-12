# source code: represents the implementation of KNN algorithm ***from the scratch***
# https://towardsdatascience.com/k-nearest-neighbors-algorithm-implementation-18b44e4ea244
# https://github.com/karankharecha/Big_Data_Algorithms/blob/master/src/knn_classification.py



import pandas as pd
import numpy as np
import plotly.offline as plt
import plotly.graph_objs as go

# define knn classifier
def knn_classifier(path, target_classes, classifier, sample, k):
    # 数据集
    training_set = pd.DataFrame(pd.read_csv(path))
    # 标签
    classes_set = training_set[target_classes]

    # draw original data graph,
    # data_graph = [go.Scatter3d(
    #     x=classes_set[target_classes[0]],
    #     # y=classes_set[target_classes[2]],
    #     # z=classes_set[target_classes[2]],
    #     mode='markers',
    #     marker=dict(
    #         color='#212121',
    #     ),
    #     name='Points from training set'
    # ), go.Scatter3d(
    #     x=[sample[0]],
    #     # y=[sample[1]],
    #     # z=[sample[2]],
    #     mode='markers',
    #     marker=dict(
    #         size=10,
    #         color='#FFD600',
    #     ),
    #     name='New sample'
    # )]
    #
    # graph_layout = go.Layout(
    #     scene=dict(
    #         xaxis=dict(title='BIO_ECG'),
    #         # yaxis=dict(title='width'),
    #         # zaxis=dict(title='height')
    #     ),
    #     margin=dict(b=10, l=10, t=10)
    # )
    #
    # data_graph = go.Figure(data=data_graph, layout=graph_layout)
    #
    # plt.plot(data_graph, filename='data/knn_classification.html')

    # euclidean distance is calculated for measuring nearness among data points.
    training_set['dist'] = (classes_set[target_classes] - np.array(sample)).pow(2).sum(1).pow(0.5)

    # A new column named ‘dist’ that contains values of all the euclidean distances from the sample,
    # is appended to the existing dataframe. Next step is about sorting the data in ascending order
    # and collect k nearest neighbouring points.
    training_set.sort_values('dist', inplace=True)
    return (training_set.iloc[:k][classifier]).value_counts().idxmax()


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    print(knn_classifier(path="data/ECG_4.csv",
                         # target_classes=['mass', 'width', 'height'],
                         target_classes=['BIO_ECG'],
                         classifier='fatiguelevel',
                         sample=[-1],
                         k=7))