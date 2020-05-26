import pandas as pd

def main():
    filename = 'data/JEC_basic_sentence_v1-2.xls' 
    df = pd.read_excel(filename, sheet_name=0)

    df.to_csv('data/csvfile.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    main()
