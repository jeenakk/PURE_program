__author__ = 'ztx'

import urllib2
from bs4 import BeautifulSoup
import nltk.data

class Parser:
    def __init__(self,iterations):

        sum = iterations;
        output=dict()
        while(iterations):
            if self.parse(output):
                iterations = iterations-1
                print(str(iterations))

        self.generatedata(output,sum)


    def parse(self,output):
        src = 'https://en.wikipedia.org/wiki/Special:Random'
        #src="http://en.wikipedia.org/wiki/Accrington_Stanley,_Who_Are_They%3F"
        read_data = urllib2.urlopen(src).read()
        self.soup = BeautifulSoup(read_data)
        #print(self.soup.get_text())
        s=""
        #todo add more paragraghs instead of the first one
        #for element in self.soup.p.contents:
        defination = self.soup.find_all('p')
        if defination.__len__()<=0:
            return False

        for element in defination:
            if(element.text!=None):
                if((" is " in element.text) or (" was " in element.text)):
                    s=s+element.text
                    break
                elif (("may refer to" in element.text) or ("Wikipedia does not have an article with this exact name" in element.text)):
                    return False
        #print(s.encode('ascii','ignore'))
        if s == "":
            return False

        title = self.soup.title.string
        title= title.encode('ascii','ignore')
        index = title.find("- Wikipedia")
        title = title[:index]
        #print(title)

        s=self.process_string(s)


        sent_detector = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
        s=sent_detector.tokenize(s.strip())[0]
        s=s.encode('ascii','ignore')


        if title in output.keys():
            return False
        else:
            output[title] = s
            return True
            #todo check duplicate in dictionary

    def process_string(self,str):
        #remove is or was and the title
        index2=str.find(" is ")
        if index2 !=-1:
            str= str[index2+4:]
        else:
            str = str[str.find(" was ")+5:]

        # remove the number and Remove parenthesis
        while(1):
            index3 = str.find("(")
            index4 = str.find(")")
            if index3 != -1 and index4 != -1:
                str = str[:index3-1] + str[index4+1:]
            else:
                break

        while(1):
            index3 = str.find("[")
            index4 = str.find("]")
            if index3 != -1 and index4 != -1:
                str = str[:index3] + str[index4+1:]
            else:
                break
        return str











    def generatedata(self,output,iterations):
        result_file = open(str(iterations)+"data.txt", "w")
        i=1
        for key,value in  output.items():
            result_file.write("%s\n" %key)
            result_file.write("%s\n\n" %value)
            i=i+1

        result_file.close()

def main():
    test = Parser(1000)





if __name__ == "__main__":
    main()