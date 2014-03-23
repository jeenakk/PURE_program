function out = getEntailmentString(num)
if( num == 1 )
    out = 'ENTAILMENT';
elseif( num == 0) 
    out = 'NEUTRAL'; 
else 
    out = 'CONTRADICTION';  
end