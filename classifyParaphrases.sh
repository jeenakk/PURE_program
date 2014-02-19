# sed '1~3d' input_entailment.txt > sentences.txt
# sed -n 'p;N;N' input_entailment.txt > labels.txt
./stanford-parser-2011-09-14/lexparser.sh sentences.txt > parsed.txt
# cd code
# echo run | matlab -nodesktop
