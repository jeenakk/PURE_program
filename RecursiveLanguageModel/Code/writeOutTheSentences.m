clear all; close all; clc; 

files = '../Data/'; 
data = textread( [files '100data.txt' ] , '%s', 'delimiter', '\n'); 
definitions = data(2:3:end);  

f = fopen('../Data/onlySentences.txt', 'w'); 
for i = 1:size(definitions,1)
    fprintf(f, [ definitions{i}  '\n'])
    % fprintf(f, '############\n' )
end 

