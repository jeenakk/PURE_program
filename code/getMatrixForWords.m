function [ matrices ] = getMatrixForWords( data )

map = getmap('mapdata.txt');
ns = size(data.allSStr);
ns = ns(2);

matrices= {};

for i=2:2:ns
    s1 = data.allSStr{i-1};
    s2 = data.allSStr{i};
    m = zeros(size(s1,2), size(s2,2));
    for j=1:1:size(s1,2)
        for k=1:1:size(s2,2)
            key=char(strcat(s1(1,j),'@',s2(1,k)));
            if(map.isKey(key))
                m(j,k) = map(key);
            else
                m(j,k) = 0;
                if(strcmp(s1(1,j), s2(1,k)))
                    m(j,k)=1;
                end
            end
        end
    end
    matrices{i/2} = m;
end

end

