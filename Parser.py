__author__ = 'ztx'
import os
import random
import sys
import math

def getdata(filename):
    if(os.path.exists(filename)):
            data=open(filename,"r")
    return data


def parse(data):
    data.next()
    result_lst = list()
    for instance in data:

        parts=instance.split("\t")
        type =parts[4].replace("\n","")
        if(type=="NEUTRAL" or type=="CONTRADICTION"):
            type=1
        elif(type=="ENTAILMENT"):
            type=0
        else:
            raise EOFError
        result_lst.append(type)
        result_lst.append(parts[1])
        result_lst.append(parts[2])
    return result_lst

def writetofile(res_lst,res_file):
    for term in result_lst:
        result_file.write("%s\n" % term)



train_file = sys.argv[1]
train_data = getdata(train_file)
result_lst = list()
if(train_data):
    result_lst=parse(train_data)
result_file = open("input_entailment.txt", "w")
writetofile(result_lst,result_file)