# -*- coding: utf-8 -*-

import xlrd
import csv
import codecs
import pandas as pd


# 将xls数据集转换为csv格式以便使用pandas工具包来进行处理
def xlsx_to_csv():
    workbook = xlrd.open_workbook('database.xls')
    table = workbook.sheet_by_index(0)
    with codecs.open('database.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)


# 对csv数据进行预处理，以得到方便进行训练的数据格式
def csv_process():
    #  按需求格式化每一列的数据，如删除A1列数据中的“A1”
    db['A1'] = db['A1'].map(lambda x: str(x)[2])
    db['A3'] = db['A3'].map(lambda x: str(x)[2])
    db['A4'] = db['A4'].map(lambda x: str(x)[2])
    db['A6'] = db['A6'].map(lambda x: str(x)[2])
    db['A7'] = db['A7'].map(lambda x: str(x)[2])
    db['A9'] = db['A9'].map(lambda x: str(x)[2])
    db['A10'] = db['A10'].map(lambda x: str(x)[3])
    db['A12'] = db['A12'].map(lambda x: str(x)[3])
    db['A14'] = db['A14'].map(lambda x: str(x)[3])
    db['A15'] = db['A15'].map(lambda x: str(x)[3])
    db['A17'] = db['A17'].map(lambda x: str(x)[3])
    db['A19'] = db['A19'].map(lambda x: str(x)[3])
    db['A20'] = db['A20'].map(lambda x: str(x)[3])
    # 保存处理后的数据为processeddata.csv文件
    db.to_csv('processeddata.csv')


# 数据截取，按信用风险评估结果和题目要求的比例进行截取
def csv_cut(db_positive_m, db_negative_m):    
    db_positive.to_csv('positivedata.csv', encoding='utf8')
    db_negative.to_csv('negativedata.csv', encoding='utf8')

# 数据合并，将截取到的用于建立模型的80%的数据合并到同一个csv文件中
def csv_merge(db_positive_m, db_negative_m):
    db_merge = pd.concat([db_positive_m, db_negative_m], axis=0)
    db_merge.to_csv('mergeddata.csv', encoding='utf8')

# 划分出用于评价模型的数据
def csv_predict(db_positive_p, db_negative_p):
    db_predict = pd.concat([db_positive_p, db_negative_p], axis=0)
    db_predict.to_csv('predictdata.csv', encoding='utf8')


if __name__ == "__main__":
    db = pd.read_csv('database.csv')
    db = pd.DataFrame(db)

    xlsx_to_csv()
    csv_process()

    db_positive = db[db.Class == 1]
    db_negative = db[db.Class == 2]
    db_positive_m = db_positive[:560]
    db_negative_m = db_negative[:240]
    db_positive_p = db_positive[560:]
    db_negative_p = db_negative[240:]

    csv_cut(db_positive_m, db_negative_m)
    csv_merge(db_positive_m, db_negative_m)
    csv_predict(db_positive_p, db_negative_p)

