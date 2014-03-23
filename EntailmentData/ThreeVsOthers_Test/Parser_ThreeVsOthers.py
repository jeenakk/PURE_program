__author__ = 'ztx'
import os
import sys


def getdata(filename):
    if(os.path.exists(filename)):
            data=open(filename,"r")
            return data
    else:
        return False


def parse(data):
    data.next()
    result_lst = list()
    for instance in data:

        parts=instance.split("\t")
        type =parts[4].replace("\n","")

        if(type=="CONTRADICTION"):
            type=-1
        elif(type=="NEUTRAL"):
            type=0
        else:
            type=1
        result_lst.append(type)
        result_lst.append(parts[1])
        result_lst.append(parts[2])
    return result_lst

def writetofile(res_lst,res_file):
    for term in res_lst:
        res_file.write("%s\n" % term)



def main():
    """
   running command has format:python Parser_ThreeVsOthers.py filename type0 type1_1 type1_2; type options are different and from NEUTRAL,ENTAILMENT,CONTRADICTION")
    """
    train_file = sys.argv[1]
    train_data = getdata(train_file)
    result_lst = list()
    if(train_data):
        result_lst=parse(train_data)
    result_file = open("ThreeVsOthers_Test.txt", "w")
    writetofile(result_lst,result_file)


if __name__ == "__main__":
    main()