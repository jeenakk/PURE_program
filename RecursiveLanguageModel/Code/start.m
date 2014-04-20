clear all; close all; clc; 
files = '../Data/'; 

data = textread( [files '100data.txt' ] , '%s', 'delimiter', '\n'); 
words = data(1:3:end); 
definitions = data(2:3:end); 

data2 = textread( [files 'tokenzied.txt' ] , '%s', 'delimiter', '\n'); 

tokenized = cell(0); 
tmp = cell(0); 
for i = 1 :size(data2 , 1)
    if ( strcmp(data2{i}, '############') ) 
        tokenized{end+1} = tmp;
        tmp = cell(0); 
    else 
        tmp{end+1} = data2{i}; 
    end 
end 


load('../../data/vars.normalized.100.mat')




words_RLM = []; 
vectors = []; 
