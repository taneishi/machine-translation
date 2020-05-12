import pandas

def read_excel(fname):
    df = pandas.read_excel(fname, sheet_name=0)
    return df

fname = 'data/JEC_basic_sentence_v1-2.xls' 
df = read_excel(fname)

df.to_csv('data/csvfile.csv', encoding='utf-8', index=False)
