import pandas

def read_excel(filename):
    df = pandas.read_excel(filename, sheet_name=0)
    return df

def main():
    filename = 'data/JEC_basic_sentence_v1-2.xls' 
    df = read_excel(filename)

    df.to_csv('data/csvfile.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    main()
