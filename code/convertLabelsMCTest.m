function out = convertLabelsMCTest(in)
output = [];
for i = 1:size(in, 2)
    if strcmp( in{i}, 'UNKNOWN' )
        output = [output 0];
    elseif strcmp(in{i}, 'ENTAILMENT')
        output = [output 1];
    end
end
    out = output; 
end