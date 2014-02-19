 # sed '1~3d' input.txt > sentences.txt
 #  sed -n 'p;N;N' input.txt > labels.txt

# ./stanford-parser-2011-09-14/lexparser.sh sentences.txt > parsed.txt

cd code
echo run | matlab -nodesktop
