function [ map ] = getmap( fname )

map= containers.Map('KeyType','char','ValueType','double');

M = importdata('mapdata.txt','@');

l=size(M.textdata);

for i=1:1:l
    s1 = M.textdata(i,1);
    s2 = M.textdata(i,4);
    d = M.data(i,3);
    map(char(strtrim(strcat(s1,'@',s2)))) = d;
end

end

