function out = tokenize(str)
dataFolder = 'ThreeVsOthers/' 
% sentences = textread(['../EntailmentData/' dataFolder 'sentences.txt'], '%s', 'delimiter', '\n'); 
sentences = textread(str, '%s', 'delimiter', '\n'); 


final=cell(0);
for i=1:size(sentences),
    res=cell(0);
    remain=sentences{i};
    while true
       [str, remain] = strtok(remain, ' ');
       if isempty(str),  break;  end
       res{end+1}=str;
    end
    final{end+1}=res;
end
out = final; 
end