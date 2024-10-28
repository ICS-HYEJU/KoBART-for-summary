import pandas as pd
import glob
import os

def concat_file():
    input_f = '/storage/hjchoi/Document_Summary_text/Training/'
    output_f = '/storage/hjchoi/Document_Summary_text/Training/nm.tsv'

    allFiles = glob.glob(input_f + '*.tsv')
    print(allFiles)

    allData = []
    for file in allFiles:
        if file.split('/')[-1] =='law.tsv':
            continue
        df = pd.read_csv(file, sep='\t')
        allData.append(df)

    dataconcat = pd.concat(allData, ignore_index=True)
    dataconcat.drop('id',axis=1,inplace=True)
    dataconcat.dropna(inplace=True)
    dataconcat.to_csv(output_f,sep='\t', index=False)

if __name__ == '__main__':
    concat_file()