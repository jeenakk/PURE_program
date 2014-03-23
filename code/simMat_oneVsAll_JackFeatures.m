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
labels = load(['../EntailmentData/' dataFolder 'labels.txt']); 
relatednessLabel = load(['../EntailmentData/' dataFolder 'relatednessLabel.txt']); 

% separate some instances for training 
% randvector = rand(size(labels));
% save('randvector', 'randvector')
load('randvector'); 

threshold = 0.25; 
randvector_doubleSize = zeros(2*length(randvector), 1); 
randvector_doubleSize(1:2:end) = randvector; 
randvector_doubleSize(2:2:end) = randvector; 

training_labels = labels( randvector > threshold ); 
testing_labels = labels( randvector<= threshold ); 

training_relatednesslabels =  relatednessLabel( randvector > threshold ); 
testing_relatednesslabels =  relatednessLabel( randvector <= threshold ); 

save('relatedness_labels', 'training_relatednesslabels', 'testing_relatednesslabels') ;

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
% commented by Daniel /home/tzhang46
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
% for i = 1:length(data.train.M)
%     data.train.M{end+1} = data.train.M{i}';
%     data.train.freq1{end+1} = data.train.freq2{i}';
%     data.train.freq2{end+1} = data.train.freq1{i}';
% end
% data.train.label = [data.train.label data.train.label];
% data.train.ldiff = [data.train.ldiff data.train.ldiff];
% data.train.numeq = [data.train.numeq data.train.numeq];
% data.train.numeq2 = [data.train.numeq2 data.train.numeq2];
% data.train.numeq3 = [data.train.numeq3 data.train.numeq3];

%%
params2.cutoff=70000;
params2.sizeM = 15;
params2.pool = 2;

[dataFullTrain,preProNorm2]= preProData_release(data.train,params2);
[dataTest]= preProData_release(data.test,params2,preProNorm2);

save(outFile, 'dataFullTrain','dataTest','params');


%% Jack's features 
% tokenized = tokenize(['../EntailmentData/' dataFolder 'sentences.txt']); 
% JackMatrices = getMatrixForWords(tokenized); 
% save('JackMatrices', 'JackMatrices');
load JackMatrices

jack_dtrain={};
jack_dtrain.M = JackMatrices(randvector > threshold);  
jack_dtrain.label = dtrain.label;
jack_dtrain.numeq = dtrain.numeq;
jack_dtrain.numeq2 = dtrain.numeq2;
jack_dtrain.numeq3 = dtrain.numeq3;
jack_dtrain.ldiff = dtrain.ldiff;    
jack_dtrain.freq1 = dtrain.freq1;
jack_dtrain.freq2 = dtrain.freq2;

jack_dtest={};
jack_dtest.M = JackMatrices(randvector <= threshold);  
jack_dtest.label = dtest.label;
jack_dtest.numeq = dtest.numeq;
jack_dtest.numeq2 = dtest.numeq2;
jack_dtest.numeq3 = dtest.numeq3;
jack_dtest.ldiff = dtest.ldiff;    
jack_dtest.freq1 = dtest.freq1;
jack_dtest.freq2 = dtest.freq2;

[ftrain,preProNorm2]= preProData_release_JackVersion(jack_dtrain,params2);
[ftest]= preProData_release_JackVersion(jack_dtest,params2,preProNorm2);

%TODO
%append X values
dataFullTrain_withJack = [dataFullTrain.X ; ftrain.X];
dataTest_withJack = [dataTest.X ;  ftest.X];

save([outFile '_withJack'], 'dataFullTrain_withJack','dataTest_withJack');

%%
%logistic regression
% clear
% clearvars -except dataFolder outFile  relatednessLabel 

clear train 

addpath('../toolbox/liblinear-1.51/matlab/');

load(outFile);

% c = [0.001 0.005 0.01 0.05 0.1 0.3 ]
c = 0.05;
% -e 
% epsilon = [0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 ]

save('upToWhenJackCame'); 

Max_acc =0;
Chosen=0;
for c =0.05  % [0.001 0.005 0.01 0.05 0.1 0.3 ]
    for e = 0.01 %  [0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 ]
        %% 0 
        labels_0 = double(dataFullTrain.labels' == 0); 
        labels_0_test = double(dataTest.labels' == 0); 
        model_0 = train(labels_0, sparse([dataFullTrain.X; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c) '  -e  '  num2str(e) ]);
        [predicted_labels_0,acc_0,dv_0] = predict(labels_0_test, sparse([dataTest.X; dataTest.otherFeat]'),model_0,'-b 1');
        sum(abs(predicted_labels_0 - (dv_0(:,1)> 0.5)) )

        %% -1 
        labels_Neg1 = double(dataFullTrain.labels' == -1); 
        labels_Neg1_test = double(dataTest.labels' == -1); 
        model_Neg1 = train(labels_Neg1, sparse([dataFullTrain.X; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c)]);
        [predicted_labels_Neg1,acc_Neg1,dv_Neg1] = predict(labels_Neg1_test, sparse([dataTest.X; dataTest.otherFeat]'),model_Neg1,'-b 1');
        sum(abs(predicted_labels_Neg1 - (dv_Neg1(:,2)> 0.5)) )

        %%  1 
        labels_1 = double(dataFullTrain.labels' == 1); 
        labels_1_test = double(dataTest.labels' == 1); 
        model_1 = train(labels_1, sparse([dataFullTrain.X; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c)]);
        [predicted_labels_1,acc_1,dv_1] = predict(labels_1_test, sparse([dataTest.X; dataTest.otherFeat]'),model_1,'-b 1');
        sum(abs(predicted_labels_1 - (dv_1(:,2)> 0.5)) )

        [MaxVal,I] = max([(dv_Neg1(:,2))';(dv_0(:,1))';(dv_1(:,2))']);
        labels_all = I-2;
        EVAL = Evaluate(dataTest.labels,labels_all); 
        disp(['Accuracy = '   num2str(EVAL(1))]); 
        disp(['F-Measure = '  num2str(EVAL(6))]);
        disp(['MSE = '  num2str(EVAL(8))]);
        temp = EVAL(1);
        if gt(temp, Max_acc)
            Max_acc = temp;
            Chosen= c;
        end
    end
end


for c =0.05  % [0.001 0.005 0.01 0.05 0.1 0.3 ]
    for e = 0.01 %  [0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 ]
        %% 0 
        labels_0 = double(dataFullTrain.labels' == 0); 
        labels_0_test = double(dataTest.labels' == 0); 
        model_0 = train(labels_0, sparse([dataFullTrain_withJack; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c) '  -e  '  num2str(e) ]);
        [predicted_labels_0,acc_0,dv_0] = predict(labels_0_test, sparse([dataTest_withJack; dataTest.otherFeat]'),model_0,'-b 1');
        sum(abs(predicted_labels_0 - (dv_0(:,1)> 0.5)) )

        %% -1 
        labels_Neg1 = double(dataFullTrain.labels' == -1); 
        labels_Neg1_test = double(dataTest.labels' == -1); 
        model_Neg1 = train(labels_Neg1, sparse([dataFullTrain_withJack; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c)]);
        [predicted_labels_Neg1,acc_Neg1,dv_Neg1] = predict(labels_Neg1_test, sparse([dataTest_withJack; dataTest.otherFeat]'),model_Neg1,'-b 1');
        sum(abs(predicted_labels_Neg1 - (dv_Neg1(:,2)> 0.5)) )

        %%  1 
        labels_1 = double(dataFullTrain.labels' == 1); 
        labels_1_test = double(dataTest.labels' == 1); 
        model_1 = train(labels_1, sparse([dataFullTrain_withJack; dataFullTrain.otherFeat]'), ['-s 0 -c ' num2str(c)]);
        [predicted_labels_1,acc_1,dv_1] = predict(labels_1_test, sparse([dataTest_withJack; dataTest.otherFeat]'),model_1,'-b 1');
        sum(abs(predicted_labels_1 - (dv_1(:,2)> 0.5)) )

        [MaxVal,I] = max([(dv_Neg1(:,2))';(dv_0(:,1))';(dv_1(:,2))']);
        labels_all = I-2;
        EVAL = Evaluate(dataTest.labels,labels_all); 
        disp(['Accuracy = '   num2str(EVAL(1))]); 
        disp(['F-Measure = '  num2str(EVAL(6))]);
        disp(['MSE = '  num2str(EVAL(8))]);
        temp = EVAL(1);
        if gt(temp, Max_acc)
            Max_acc = temp;
            Chosen= c;
        end
    end
end


%%  Using Singer's Multi-Class classifier 
model_s_4 = train(dataFullTrain.labels', sparse([dataFullTrain.X; dataFullTrain.otherFeat]'), ['-s 4 -c ' num2str(c)]);
[predicted_labels_s_4,acc_s_4] = predict(dataTest.labels', sparse([dataTest.X; dataTest.otherFeat]'),model_1);
EVAL = Evaluate(dataTest.labels',predicted_labels_s_4); 
disp(['Accuracy = '   num2str(EVAL(1))]); 
disp(['F-Measure = '  num2str(EVAL(6))]);
disp(['MSE = '  num2str(EVAL(8))]);

%% RelatedNess : 
load 'relatedness_labels'
input_train = [dataFullTrain.X; dataFullTrain.otherFeat]'; 
input_test = [dataTest.X; dataTest.otherFeat]'; 
% training_relatednesslabels 
% [row_size,col_size]=size(input_train);
% R=cell(0);
% 
% for i = 1:col_size
%     
%     
%  
% end
% R = [];
% S = [5,1];

% net = newff(input_train', training_relatednesslabels', [1] , {'tansig'}, 'trainlm');
Min = -1;
Method=0;
layers1=0;
layers2=0;
for k = 1:15
    k 
    for i = 1:15
        i 
        for j=1:2
            j 
            if eq(j,1)
                net = newff(input_train', training_relatednesslabels', [k i] , {'tansig'}, 'trainlm');
            else
                net = newff(input_train', training_relatednesslabels', [k i] , { 'purelin'}, 'trainlm');
            end
            net.trainParam.showWindow = false; 
            net = train(net,input_train',training_relatednesslabels');
            similarityPrediction = 4*sim(net, input_test')+1; 
            temp= mean( (testing_relatednesslabels' - similarityPrediction ).^2 );
            if eq(Min,-1)
                Min = temp;
            elseif ge(Min,temp)
                Min = temp
                Method=j;
                layers1=k;
                layers2=i;
                minNet = net; 
            end
        end
    end
end
% best way Min = 6.2227, Method = tansig and layers =4
% size hidden layer  (1 to 15)
% tansig purelin 
% number of hidden layers (1 to 2)
% view(net)
% net = train(net,input_train',training_relatednesslabels');
% similarityPrediction = 4*sim(net, input_test')+1; 
% 
% mean( (testing_relatednesslabels' - similarityPrediction ).^2   )
% 
% [RHO_s,PVAL_s] = corr(testing_relatednesslabels',similarityPrediction,'Type','Spearman');
% [RHO_p,PVAL_p] = corr(testing_relatednesslabels',similarityPrediction,'Type','Pearson');
% 
% disp('Spearmans rho : ')
% RHO_s
% PVAL_s
% disp('Pearsons rho : ')
% RHO_p
% PVAL_p

pairId_train = load(['../EntailmentData/PairId_train.txt']); 
writeTheSemEVALOutput('') 

% dlmwrite([ '../EntailmentData/' dataFolder 'output.txt'],predicted_labels)

% EVAL = Evaluate(dataTest.labels,predicted_labels'); 
% disp(['Accuracy = '   num2str(EVAL(1))]); 
% disp(['Sensitivity = '  num2str(EVAL(2))]); 
% disp(['Specificity = '  num2str(EVAL(3))]); 
% disp(['Precision = '  num2str(EVAL(4))]); 
% disp(['Recall = '  num2str(EVAL(5))]); 
% disp(['F-Measure = '  num2str(EVAL(6))]);
% disp(['G-mean = '  num2str(EVAL(7))]);
% disp(['MSE = '  num2str(EVAL(8))]);
% 


