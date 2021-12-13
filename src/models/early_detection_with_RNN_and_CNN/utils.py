import sys, os, fnmatch, datetime, time
import sqlite3
import numpy as np
from scipy import misc
from sklearn import preprocessing
import ast

def read(file_path):
    if not os.path.exists(file_path):
        return None
    f = open(file_path, 'r', encoding = 'utf-8', errors = 'ignore')
    con = f.read()
    f.close()
    con = ast.literal_eval(con)
    con = [n.strip() for n in con]
    return con
    
def read_lines(file_path):
    if not os.path.exists(file_path):
        return None
    f = open(file_path, "r", encoding='utf-8', errors = 'ignore')
    ls = f.readlines()
    ls2 = []
    for line in ls:
        if line.strip():
            ls2.append(line.strip())
    f.close()
    return ls2

def save(var, file_path):
    f = open(file_path, "w", encoding='utf-8', errors = 'ignore')
    f.write(str(var))
    f.close()
    

def ls(folder, pattern = '*'):
    fs = []
    for root, dir, files in os.walk(folder):
        for f in fnmatch.filter(files, pattern):
            fs.append(f)
    return fs

def str_to_timestamp(string):
    structured_time = time.strptime(string,"%a %b %d %H:%M:%S +0000 %Y")
    timestamp = time.mktime(structured_time)
    return timestamp

def db_connect(db_file):
    try:
        dbc = sqlite3.connect(db_file, timeout=10)
        return dbc
    except lite.Error as e:   
        print ("Error {}:".format(e.args[0]))
        return None

def normalize(x,positions):
    num_columns = x.shape[1]
    for i in range(num_columns):
        if i in positions:
            x[:,i:i+1] = np.copy(preprocessing.robust_scale(x[:,i:i+1]))
    return x

def load_img(path,length,height):
    img = misc.imread(path)
    img = misc.imresize(img,[height,length])
    img = img[:,:,0:3]
    return img
