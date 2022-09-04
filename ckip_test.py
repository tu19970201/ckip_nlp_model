
import pandas as pd
import os
from ckiptagger import construct_dictionary, WS, POS, NER

# 下載斷詞模組
# from ckiptagger import data_utils
# data_utils.downlaod_data_gdown('./')

def import_stopwords():
    # 匯入stopwords
    with open('stopwords.csv','r',encoding='utf-8') as file:
        stopword_file = pd.read_csv(file, header=None)
    stopwords_list = stopword_file.values.tolist()
    
    stopwords = set()
    for s in stopwords_list:
        for word in s:
            if pd.isnull(word) == False:
                if type(word) == float:
                    word = str(int(word))
                stopwords.add(word)
    print("import stopwords")

    return list(stopwords)

def import_customized():
    # 匯入自定斷詞
    with open('customized.csv','r',encoding='utf-8') as file:
        customized_file = pd.read_csv(file, header=None)
    customized_list = customized_file.values.tolist()
    
    customized = set()
    for c in customized_list:
        for word in c:
            if pd.isnull(word) == False:
                if type(word) == float:
                    word = str(int(word))
                customized.add(word)
                
    print("import customized")
    print('=' * 30)
    customized.update(stopwords)
    return list(customized)

def word_to_weight():
    # 給斷詞權重   
    word_to_weight = {}
    for c in customized:
        word_to_weight.update({c : 1})
        
    dictionary = construct_dictionary( word_to_weight )
    return dictionary


if __name__ == "__main__":
    
    stopwords = import_stopwords()
    customized = import_customized()
    dictionary = word_to_weight()
    
    
    path = 'labeled_data'
    file_names = os.listdir(path)
    
    file_num = 0
    for file_name in file_names:
        
        # 讀取資料
        with open(f'{path}/{file_name}','r',encoding='utf-8') as ptt_file:
            pt_data = pd.read_csv(ptt_file, header=None)
        text : list = pt_data[1].values.tolist()
    
        #斷詞
        ws = WS("./data") 
        # ws_results = ws(text,recommend_dictionary=dictionary)
        ws_results = ws(text,coerce_dictionary=dictionary)
        
        # 去掉stopwords
        outstr = []
        for index, sentence in enumerate(ws_results) :
            outstr.append([])
            for word in sentence:
                if word not in stopwords :
                    outstr[index].append(word)
    
        # 將label與斷詞結果結合
        label : list = pt_data[0].values.tolist()
        results = pd.DataFrame(list(zip(label, outstr)), columns=['label', 'outstr']).dropna()
        results = results[results['outstr'].map(lambda d: len(d)) > 0]
    
        # 存入csv
        results.to_csv(f'ws_data/{file_name[:10]}_ws.csv', index = False)
        
        file_num += 1
        print('=' * 30)
        print(f'{file_name} is cleaned up')
        print(f'{file_num}/{len(file_names)}')
        print('=' * 30)




    
