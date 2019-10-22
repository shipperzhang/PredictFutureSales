import os
import gzip

def loadGzipData(path):
    f=gzip.open(path,'r')
    file_content=f.read()
    data = file_content.split()
    print(data[0])
    return data

def loadCSVData(path):
    pass


if __name__ == "__main__":
    loadGzipData('data/test.csv.gz')