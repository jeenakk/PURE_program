% load the data 
ContraVsOthers = '../EntailmentData/ContradictionVsOthers/' ;
EntailVsOthers = '../EntailmentData/EntailmentVsOthers/' ;
NeutralVsOthers = '../EntailmentData/NeutralVsOthers/' ;
% dataFolder = 'SimilarityMeasure/'; 

load([ContraVsOthers 'output.txt' ])
ContraVsOthers_output = output; 
load([EntailVsOthers 'output.txt' ])
EntailVsOthers_output = output; 
load([NeutralVsOthers 'output.txt' ])
NeutralVsOthers_output = output; 

% ContraVsOthers_output(end-5:end)
% EntailVsOthers_output(end-5:end)
% NeutralVsOthers_output(end-5:end)


load([ContraVsOthers 'labels.txt' ])
ContraVsOthers_label = labels; 
load([EntailVsOthers 'labels.txt' ])
EntailVsOthers_label = labels; 
load([NeutralVsOthers 'labels.txt' ])
NeutralVsOthers_label = labels; 

sum(ContraVsOthers_label  .* EntailVsOthers_label  .* NeutralVsOthers_label )

% 0 : NEUTRAL 
% 1 : ENT 
% -1: CONT

% EntailmentVsOthers 
% ContradictVsOthers 
LABEL1 = zeros(size(EntailVsOthers_label )); 
LABEL1(EntailVsOthers_label==0) = 1; 
LABEL1(ContraVsOthers_label==0) = -1; 

% EntailmentVsOthers 
% NeutralVsOthers 
LABEL2 = -1 * ones(size(EntailVsOthers_label)); 
LABEL2(EntailVsOthers_label==0) = 1; 
LABEL2(NeutralVsOthers_label ==0) = 0; 

% ContradictVsOthers 
% EntailmentVsOthers 
LABEL3 = zeros(size(EntailVsOthers_label)); 
LABEL3(ContraVsOthers_label==0) = -1; 
LABEL3(EntailVsOthers_label==0) = 1; 

% ContradictVsOthers 
% NeutralVsOthers 
LABEL4 = ones(size(EntailVsOthers_label)); 
LABEL4(ContraVsOthers_label==0) = -1; 
LABEL4(NeutralVsOthers_label==0) = 0; 


sum(sum(LABEL1 - LABEL2))
sum(sum(LABEL3 - LABEL4))


