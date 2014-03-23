function writeTheSemEVALOutput( outputName, predictedRelatedNess, predictedEntailment, pairID )
fileID = fopen(outputName, 'w');
fprintf(fileID, 'pair_ID	relatedness_score	entailment_judgment\n ');

for i = 1: size( pairID, 1 )
    if( isempty(predictedRelatedNess) &&  isempty(predictedEntailment)  )
        fprintf(fileID, '%d    NaN     NaN \n', pairID(i) );
    elseif ( isempty(predictedRelatedNess)  )
        fprintf(fileID, '%d   NaN     %s\n', pairID(i) , getEntailmentString(  predictedEntailment(i) ));
    elseif( isempty(predictedEntailment)  )
        fprintf(fileID, '%d    %f    NaN\n', pairID(i) , predictedRelatedNess(i));
    else     
        fprintf(fileID, '%d	%f	%s\n', pairID(i) , predictedRelatedNess(i), getEntailmentString(  predictedEntailment(i) ) );
    end
end


