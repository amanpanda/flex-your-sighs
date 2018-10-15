import pandas as pd
import os

def preprocess_aoa():
    data_aoa = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/AoA_51715_words.csv'),
                       header=0, delimiter=",", encoding='utf-8')
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'datasets/Freq_Lexicons.csv'),
                           header=0, delimiter=",", encoding='utf-8')
    l = list(data_aoa['Word'])
    aoa = []
    for word in data['_Word']:
        if word in l:
            aoa.append(data_aoa['AoA_Kup_lem'][l.index(word)])
        else:
            aoa.append('NaN')
    data['aoa'] = aoa
    data.to_csv('datasets/Freq_aoa_lexicons.csv')
