function index = WordLookup(InputString)
global wordMap
if wordMap.isKey(InputString)
    index = wordMap(InputString);
else
    wordMap(InputString) = wordMap('*UNKNOWN*'); 
    % index=wordMap('*UNKNOWN*');
end
