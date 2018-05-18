import os
import pickle
import math
import numpy as np
import json

def get_substrings(string):
    output=[]
    for i in range(len(string)):
        for j in range(i+1,len(string)+1):
            output.append(string[i:j])
    return output
            
def find_more_similar(string1,list_string):
    best_cont=0
    best_string=''
    substrings=get_substrings(string1)
    for string2 in list_string:
        cont=0
        for string in substrings:
            if string.lower() in string2.lower() and len(string)>cont:
#                 print(string,string2)
                cont=len(string)
        if cont>best_cont:
            best_cont=cont
            best_string=string2
    return best_string

def find_integer_end_string(string):
    count=-1
    while string[count].isdigit():
        count-=1
    return string[count+1:]

def euclidean_distance(pair1,pair2):
    return math.sqrt((pair1[0]-pair2[0])**2+(pair1[1]-pair2[1])**2)

def remove_array_in_list(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def read_json(file_json,extension='json'):
    if not file_json.endswith('.json'):
        file_json+='.json'
    with open(file_json) as f:
        data=json.load(f)
    return data

def read_pickle(file,extension='pickle'):
    if '.' not in file:
        file+='.'+extension
    with open(file, 'rb') as handle:
        return pickle.load(handle)

def write_file(variable,output_name):
    file=open(output_name,'w')
    file.write(variable)
    file.close()

def write_pickle(variable,output_name):
    if '.' not in output_name:
        output_name+='.pickle'
    with open(output_name, 'wb') as handle:
        pickle.dump(variable, handle )


def readlines_file(path):
    f=open(path)
    lines=f.readlines()
    f.close()

    return lines


