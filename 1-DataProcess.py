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


if __name__ == "__main__":
    db = pd.read_csv('database.csv')
    # db = db.drop(['A9', 'A20'], axis=1)
    db = pd.DataFrame(db)

    xlsx_to_csv()
    csv_process()

