addpath(genpath('../toolbox/'));

% load trained RAE
load('../savedParams/params.mat');

outFile = ['../savedParams/' dataFolder 'simMat_release.mat']; 
% '../savedParams/simMat_release.mat';

% Load files
load('../data/vars.normalized.100.mat');   % word representations
% fid = fopen(['../EntailmentData/' dataFolder 'sentences.txt'], 'r');
% sentences = fscanf(fid, '%s'); %
sentences = textread(['../EntailmentData/' dataFolder 'sentences.txt'], '%s', 'delimiter', '\n'); 

load([ '../EntailmentData/' dataFolder 'test.mat']);
labels = load(['../EntailmentData/' dataFolder 'labels.txt']); %

% separate some instances for training 
randvector = rand(size(labels)); 
threshold = 0.25; 
randvector_doubleSize = zeros(2*length(randvector), 1); 
randvector_doubleSize(1:2:end) = randvector; 
randvector_doubleSize(2:2:end) = randvector; 

training_labels = labels( randvector > threshold ); 
testing_labels = labels( randvector<= threshold ); 

train.allSNum = test.allSNum( randvector_doubleSize > threshold );  
train.allSStr = test.allSStr( randvector_doubleSize > threshold );  
train.allSTree = test.allSTree( randvector_doubleSize > threshold );  
train.allSKids = test.allSKids( randvector_doubleSize > threshold );  

test1.allSNum = test.allSNum( randvector_doubleSize <= threshold );  
test1.allSStr = test.allSStr( randvector_doubleSize <= threshold );  
test1.allSTree = test.allSTree( randvector_doubleSize <= threshold );  
test1.allSKids = test.allSKids( randvector_doubleSize <= threshold );  

test = test1; 

% load('../data/msrp_all.mat');              % msrp dataset
% converted msrp dataset
% contains:
%   allSNum: array of each word's index in the dictionary
%   allSStr: array of words
%   allSTree: tree structure. allSTree[i] = j means j is i's parent
%   allSKids: children info. of the tree.
%             allSKids[i,1] is the i's left child
%             allSKids[i,2] is the i's right child
% load('../data/msrp_train_binarized_unq.100.mat'); 
% load test sentences and labels
% commented by Daniel 
% load([ '../EntailmentData/' dataFolder 'test.mat']);
% testing_labels = load(['../EntailmentData/' dataFolder 'labels.txt']); %

load('../data/MSRWordCounter'); 

load('../data/both_binarized_unq.100.mat');
global wordCounter;
wordCounter = containers.Map(keys,values);

% compute features
[Trees1 Trees2 Trees3 Trees4] = getInstancesParsed(train.allSNum,train.allSKids,train.allSTree,test.allSNum,test.allSKids,test.allSTree,theta,We2,params);


%% compute similarity matrix
type = 'eucdist';
pl = getDistances(Trees1,Trees2,type);
plt = getDistances(Trees3,Trees4,type);


%%
training_sentences = sentences(randvector_doubleSize> threshold);
trains1 = training_sentences(1:2:end);
trains2 = training_sentences(2:2:end);

testing_sentences = sentences(randvector_doubleSize <= threshold);
tests1 = testing_sentences(1:2:end);
tests2 = testing_sentences(2:2:end);


trainfreq1 = cell(length(pl),1);
trainfreq2 = cell(length(pl),1);

testfreq1 = cell(length(plt),1);
testfreq2 = cell(length(plt),1);


for i=1:length(pl)
    trainfreq1{i} = zeros(size(Trees1{i}.pp));
    for j = 1:size(Trees1{i}.pp,2)
        trainfreq1{i}(j) = countLookup(int2str(Trees1{i}.nums(Trees1{i}.ngrams(j,1):Trees1{i}.ngrams(j,2))));
    end
    
    trainfreq2{i} = zeros(size(Trees2{i}.pp));
    for j = 1:size(Trees2{i}.pp,2)
        trainfreq2{i}(j) = countLookup(int2str(Trees2{i}.nums(Trees2{i}.ngrams(j,1):Trees2{i}.ngrams(j,2))));
    end
end

for i=1:length(plt)
    testfreq1{i} = zeros(size(Trees3{i}.pp));
    for j = 1:size(Trees3{i}.pp,2)
        testfreq1{i}(j) = countLookup(int2str(Trees3{i}.nums(Trees3{i}.ngrams(j,1):Trees3{i}.ngrams(j,2))));
    end
    
    testfreq2{i} = zeros(size(Trees4{i}.pp));
    for j = 1:size(Trees4{i}.pp,2)
        testfreq2{i}(j) = countLookup(int2str(Trees4{i}.nums(Trees4{i}.ngrams(j,1):Trees4{i}.ngrams(j,2))));
    end
end

%% compute other features
dtrain = {};
dtest = {};

for i=1:length(pl)
    t7 = Trees1{i};
    t8 = Trees2{i};

    ldiff = abs((length(t7.pp)+1)/2-(length(t8.pp)+1)/2);

    s1 = trains1{i};
    s2 = trains2{i};
    x = regexp(s1,'[-+]?[0-9]*\.?[0-9]+','match');
    y = regexp(s2,'[-+]?[0-9]*\.?[0-9]+','match');

    iXY = intersect(x,y);
    numeq = isequal(iXY, union(x,y));
    
    if ~isempty(x) && ~isempty(y) && numeq
        numeq2 = 1;
    else
        numeq2 = 0;
    end
    
    if isequal(sort(x),iXY) || isequal(sort(y),iXY)
        numeq3 = 1;
    else
        numeq3 = 0;
    end
        
    dtrain.M{i} = pl{i};
    dtrain.label(i) = training_labels(i);
    dtrain.numeq(i) = numeq;
    dtrain.numeq2(i) = numeq2;
    dtrain.numeq3(i) = numeq3;
    dtrain.ldiff(i) = ldiff;
    
    dtrain.freq1{i} = trainfreq1{i};
    dtrain.freq2{i} = trainfreq2{i};
end
    
for i=1:length(plt)
    t1 = Trees3{i};
    t2 = Trees4{i};

    ldiff = abs((length(t1.pp)+1)/2-(length(t2.pp)+1)/2);

    s1 = tests1{i};
    s2 = tests2{i};
    x = regexp(s1,'[-+]?[0-9]*\.?[0-9]+','match');
    y = regexp(s2,'[-+]?[0-9]*\.?[0-9]+','match');
    
    iXY = intersect(x,y);
    numeq = isequal(iXY, union(x,y));
    
    if ~isempty(x) && ~isempty(y) && numeq
        numeq2 = 1;
    else
        numeq2 = 0;
    end
    
    if isequal(sort(x),iXY) || isequal(sort(y),iXY)
        numeq3 = 1;
    else
        numeq3 = 0;
    end
    
    dtest.M{i} = plt{i};
    dtest.label(i) = testing_labels(i);
    dtest.numeq(i) = numeq;
    dtest.numeq2(i) = numeq2;
    dtest.numeq3(i) = numeq3;
    dtest.ldiff(i) = ldiff;
    
    dtest.freq1{i} = testfreq1{i};
    dtest.freq2{i} = testfreq2{i};
end

data.train = dtrain;
data.test = dtest;


%%
% add transpose of sim. matrix
for i = 1:length(data.train.M)
    data.train.M{end+1} = data.train.M{i}';
    data.train.freq1{end+1} = data.train.freq2{i}';
    data.train.freq2{end+1} = data.train.freq1{i}';
end
data.train.label = [data.train.label data.train.label];
data.train.ldiff = [data.train.ldiff data.train.ldiff];
data.train.numeq = [data.train.numeq data.train.numeq];
data.train.numeq2 = [data.train.numeq2 data.train.numeq2];
data.train.numeq3 = [data.train.numeq3 data.train.numeq3];

%%
params2.cutoff=70000;
params2.sizeM = 15;
params2.pool = 2;

[dataFullTrain,preProNorm2]= preProData_release(data.train,params2);
[dataTest]= preProData_release(data.test,params2,preProNorm2);

save(outFile, 'dataFullTrain','dataTest','params');

%%
%logistic regression
% clear
clearvars -except dataFolder outFile 

addpath('../toolbox/liblinear-1.51/matlab/');

load(outFile);

c = 0.05;

model = train(dataFullTrain.labels', sparse([dataFullTrain.X; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c)]);
[predicted_labels,acc,dv] = predict(dataTest.labels', sparse([dataTest.X; dataTest.otherFeat]'),model,'-b 1');
dlmwrite([ '../EntailmentData/' dataFolder 'output.txt'],predicted_labels)
EVAL = Evaluate(dataTest.labels,predicted_labels'); 
disp(['Accuracy = '   num2str(EVAL(1))]); 
disp(['Sensitivity = '  num2str(EVAL(2))]); 
disp(['Specificity = '  num2str(EVAL(3))]); 
disp(['Precision = '  num2str(EVAL(4))]); 
disp(['Recall = '  num2str(EVAL(5))]); 
disp(['F-Measure = '  num2str(EVAL(6))]);
disp(['G-mean = '  num2str(EVAL(7))]);
