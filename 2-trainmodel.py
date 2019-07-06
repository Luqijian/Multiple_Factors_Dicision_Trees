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


if __name__ == '__main__':
        X_train, X_test, y_train, y_test, vec = data_Preprocess() 
        build_tree(X_train, X_test, y_train, y_test, vec)

