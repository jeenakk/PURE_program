PURE_Program
============


TODO: 
============

Parser.py: Adding options for generating data with the following cases: 

1) ENTAILMENT  vs. ( NEUTRAL or CONTRADICTION) -> already done. 

2) CONTRADICTION  vs. ( NEUTRAL or ENTAILMENT) 

3) NEUTRAL  vs. ( CONTRADICTION or ENTAILMENT) 

4) Similarity measure (scaled to one)


Generate and parse all data into their corresponding folders inside 'EntailmentData/'



Wiki of the results
===========
https://wiki.engr.illinois.edu/display/~khashab2/RTE+project

Data
============
The data is classified inside 'EntailmentData/'

../data/vars.normalized.100.mat : Contains the word vectors 


What is the difference between 'params.mat' and 'simMat_release.mat'


Variables 
============
Meaning of some important variables: 


TODO: 


allSNum: array of each word's index in the dictionary

allSStr: array of words

allSTree: tree structure. allSTree[i] = j means j is i's parent

allSKids: children info. of the tree.

          allSKids[i,1] is the i's left child

          allSKids[i,2] is the i's right child

allSOStr = {};

allSPOS = {};


