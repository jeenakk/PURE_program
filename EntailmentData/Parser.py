__author__ = 'ztx'
import os
import sys


def getdata(filename):
    if(os.path.exists(filename)):
            data=open(filename,"r")
            return data
    else:
        return False


def parse(data,type0,type1_1,type1_2):
    data.next()
    result_lst = list()
    for instance in data:

        parts=instance.split("\t")
        type =parts[4].replace("\n","")
        if(type==type1_1 or type==type1_2):
            type=1
        elif(type==type0):
            type=0
        else:
            raise EOFError
        result_lst.append(type)
        result_lst.append(parts[1])
        result_lst.append(parts[2])
    return result_lst

def writetofile(res_lst,res_file):
    for term in res_lst:
        res_file.write("%s\n" % term)

def checktype(type):
    if(type=="NEUTRAL" or type=="ENTAILMENT" or type=="CONTRADICTION"):
        return True
    else:
        return False

def main():
    """
   running command has format:python Parser.py filename type0 type1_1 type1_2; type options are different and from NEUTRAL,ENTAILMENT,CONTRADICTION")
    """
    train_file = sys.argv[1]
    type0=sys.argv[2]
    type1_1=sys.argv[3]
    type1_2 = sys.argv[4]
    if(checktype(type0) and checktype(type1_1) and checktype(type1_2)):
        train_data = getdata(train_file)
        result_lst = list()
        if(train_data):
            result_lst=parse(train_data,type0,type1_1,type1_2)
        result_file = open(type0+"VsOthers.txt", "w")
        writetofile(result_lst,result_file)
    else:
       print("three different option types can only be NEUTRAL,ENTAILMENT,CONTRADICTION!")

if __name__ == "__main__":
    main()