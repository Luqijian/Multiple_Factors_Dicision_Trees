import csv
import pydot, pydotplus
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# 对数据进行预处理，转化为矩阵形式
def data_Preprocess():
        # 从csv文件建立用于训练的数据集
        dataset = open('processeddata.csv', 'r', encoding='utf8')
        dataset = csv.reader(dataset)
        headers = next(dataset)
        # 将每一行的数据按字典的形式存入列表
        featureList = []
        labelList = []
        for row in [rows for rows in dataset]:
                labelList.append(row[len(row) - 1])
                rowDict = {}
                for i in range(1, len(row) - 1):
                        rowDict[headers[i]] = row[i]
                featureList.append(rowDict)

        # 将原始数据转化为矩阵
        vec = DictVectorizer()
        dummyX = vec.fit_transform(featureList).toarray()

        # print('dummyX:' + str(dummyX))
        # print(vec.get_feature_names())
        # print('labellist:' + str(labelList))

        # 将要预测的列转化为数组
        lb = preprocessing.LabelBinarizer()
        dummyY = lb.fit_transform(labelList)
        # print('dummyY:' + str(dummyY))

                
        X_train, X_test, y_train, y_test= train_test_split(dummyX, dummyY, test_size=0.2, random_state=0)

        return X_train, X_test, y_train, y_test, vec


# 生成决策树并进行图形绘制
def build_tree(X_train, X_test, y_train, y_test, vec):
        # 创建决策树
        clf = tree.DecisionTreeClassifier(class_weight='balanced', criterion='gini', min_samples_split=4)  # 指明为ID3算法,信息增益设置为熵(entropy)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        score1 = clf.score(X_train, y_train)

        print('clf:' + str(clf))
        print(score)
        print(score1)
        # 直接导出为pdf树形结构
        dot_data = StringIO()
        tree.export_graphviz(clf, out_file=dot_data)
        # graph = pydot.graph_from_dot_data(dot_data.getvalue())
        # graph[0].write_pdf("iris.pdf")
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("TheTree.pdf")

        with open('allDataInformationGainOri.dot', 'w') as f:
                f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
        # 通过命令行dot -Tpdf allDataInformationGainOri.dot -o output.pdf 输出树形结构
        # return clf


# # 准备用于评价的数据集
# def pdata_Preprocess():
#         dataset = open('predictdata.csv', 'r', encoding='utf8')
#         dataset = csv.reader(dataset)
#         headers = next(dataset)

#         # 将每一行的数据按字典的形式存入列表
#         featureList = []
#         labelList = []
#         for row in [rows for rows in dataset]:
#                 labelList.append(row[len(row) - 1])
#                 rowDict = {}
#                 for i in range(1, len(row) - 1):
#                         rowDict[headers[i]] = row[i]
#                 featureList.append(rowDict)

#         # 将原始数据转化为矩阵
#         vec = DictVectorizer()
#         dummyX = vec.fit_transform(featureList).toarray()

#         # 将要预测的列转化为数组
#         lb = preprocessing.LabelBinarizer()
#         dummyY = lb.fit_transform(labelList)

#         return dummyX, dummyY


# # 使用百分之二十的数据集对模型准确性进行评价
# def tree_predict(dummyX_p, dummyY_p, clf_p):
#         # # 预测数据
#         # one = dummyX_p[2, :]
#         # print('one' + str(one))
#         # # one输出为one[1. 0. 0. 0. 1. 1. 0. 0. 1. 0.]
#         # # 上面的数据对应下面列表：
#         # # ['age=middle_aged', 'age=senior', 'age=youth', 'credit_rating=excellent', 'credit_rating=fair', 'income=high', 'income=low', 'income=medium', 'student=no', 'student=yes']

#         # # 设置新数据
#         # new = one
#         # new[4] = 1
#         # new[3] = 0
#         # predictedY = clf_p.predict(new.reshape(-1, 10))  # 对新数据进行预测
#         # print('predictedY:' + str(predictedY))  # 输出为predictedY:[1]，表示愿意购买，1表示yes
#         score = clf_p.score(dummyX_p, dummyY_p)
#         print(score)



if __name__ == '__main__':
        X_train, X_test, y_train, y_test, vec = data_Preprocess()
        # clf_p = 
        build_tree(X_train, X_test, y_train, y_test, vec)
        # dummyX_p, dummyY_p = pdata_Preprocess()
        # tree_predict(dummyX_p, dummyY_p, clf_p)

