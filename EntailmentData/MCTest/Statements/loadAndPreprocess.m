% name = 'mc1600x2Edev0x2Estatements'; 
% name = 'mc1600x2Etest0x2Estatements'; 
% name = 'mc1600x2Etrain0x2Estatements'; 
% name = 'mc5000x2Edev0x2Estatements'; 
% name = 'mc5000x2Etest0x2Estatements'; 
name = 'mc5000x2Etrain0x2Estatements'; 
values = eval(name);

text = {}; 
hypothesis = {}; 
decision = {}; 

for i = 2:4:(size(eval(name),1)-1)
   
    splitted = regexp(values{i}, '"', 'split')
    decision{end+1} = splitted(4); 
    
    textTmp = regexp(values{i+1}, '[><]', 'split'); 
    text{end+1} = textTmp(3); 
    
    hypothesisTmp = regexp(values{i+2}, '[><]', 'split')
    hypothesis{end+1}  = hypothesisTmp(3); 
%     values{i+3}
end
save(name, 'decision', 'text', 'hypothesis')

