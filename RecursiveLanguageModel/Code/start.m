clear all; close all; clc; 

files = '../Data/'; 
inputFile = [files '100parsed.txt' ]; 
outputDataFile= [files '/100allParsed.mat'];
% load('../../data/vars.normalized.100.mat')
load('../Data/pretrainedWeights.mat')

if 1 
    clear words 
    load([files '100allParsed.mat' ]) 
else 
    convertStanfordParserTrees2; 
end 

data = textread( [files '100data.txt' ] , '%s', 'delimiter', '\n'); 
titles = data(1:3:end); 
definitions = data(2:3:end); 

% add the new words in the titles 
for i = 1:size(titles)
    if ~wordMap.isKey(titles{i})
        wordMap(titles{i}) = wordMap.Count + 1;
        words{end+1} = titles;
    end
end 

% add new vectors 
Wv = [Wv repmat(Wv(:, wordMap('UNK')), 1, wordMap.Count - size(Wv, 2)) ]; % repeat unknown !  

targetVectors = []; 
for i = 1:size(titles)
    targetVectors = [ targetVectors  Wv(:, wordMap(titles{i})) ]; 
end


% data2 = textread( [files 'tokenzied.txt' ] , '%s', 'delimiter', '\n'); 
% tokenized = cell(0); 
% tmp = cell(0); 
% for i = 1 :size(data2 , 1)
%     if ( strcmp(data2{i}, '############') ) 
%         tokenized{end+1} = tmp;
%         tmp = cell(0); 
%     else 
%         tmp{end+1} = data2{i}; 
%     end 
% end 
% words_RLM = []; 
% vectors = []; 

[params options] = initParams(); 

% Init Dual Parameters
Wo = 0.01*randn(params.wordSize + 2*params.wordSize*params.rankWo,length(words));
Wo(1:params.wordSize,:) = ones(params.wordSize,size(Wo,2));
% Wcat = randWcat(params);

% matlab pooling
% success = false;
% for i = 1:5
%     try
%         if ~ismac && isunix && matlabpool('size') == 0 && (~isfield(options,'DerivativeCheck') || (isfield(options,'DerivativeCheck') && strcmpi(options.DerivativeCheck,'off')))
%             numCores = feature('numCores')
%             if numCores==16
%                 numCores=8
%             end
%             matlabpool('open',numCores);
%         end
%         success=true;
%         break;
%     catch err
%         display(['Error: ' err.message ' retrying...']);
%     end
% end
% if ~success
%     display('Retries unsuccesfull');
%     rethrow(err);
% end

if 1 
    sentences = 1:10; 
else 
    sentences = 1:size(allSNum); 
end 

% TRAIN
% [allSNum_batch, Wv_batch, Wo_batch, allWordInds, params] = ...
%     getRelevantWords(allSNum,allSNN,sentences,Wv,Wo,params);
[allSNum_batch, Wv_batch, Wo_batch, allWordInds, params] = ...
    getRelevantWords(allSNum,sentences,Wv,Wo,params);

% test gettting a cost value 
% [X decodeInfo] = param2stack(Wv_batch,Wo_batch,W,WO);
% costFct_preTrainDual( X ,decodeInfo,params,allSNum_batch,allSStr,allSTree,allSNN_batch,sentenceLabels);
% [cost,grad]  = costFct_preTrainDual( X ,decodeInfo,params,allSNum_batch,allSStr,allSTree, targetVectors); 

% Optimize
[X decodeInfo] = param2stack(Wv_batch,Wo_batch,W,WO);
X = minFunc(@costFct_preTrainDual,X,options,decodeInfo,params,allSNum_batch,allSStr,allSTree, targetVectors);
[Wv_batch2,Wo_batch2,W2,WO2] = stack2param(X, decodeInfo);

% map vocab back
Wv(:,allWordInds) = Wv_batch2;
Wo(:,allWordInds) = Wo_batch2;


