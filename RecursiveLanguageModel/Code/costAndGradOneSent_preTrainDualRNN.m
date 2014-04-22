function [df_s_Wv,df_s_Wo,df_s_W,df_s_WO,cost] = costAndGradOneSent_preTrainDualRNN(sNum,sTree,sStr,Wv,Wo,W,WO,targetVector, params)
% function [df_s_Wv,df_s_Wo,df_s_W,df_s_WO,cost] = costAndGradOneSent_preTrainDualRNN(sNum,sTree,sStr,sNN,Wv,Wo,W,WO,targetVector, params)

% save('tmp5')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward Prop: Greedy Tree Parse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tree = forwardPropTree(sNum,sTree,sStr,sNN,indicies,Wv,Wo,W,WO, params);
tree = forwardPropTree(sNum,sTree,sStr,Wv,Wo,W,WO, params);

% cost = -log(tree.y(label));
cost = 0; 
predicted = tree.y;
% disp('Predicted = ')
% for i = 1:size(targetVector)
%     disp([ num2str(predicted(i))  '   '])
% end
% disp('\nTarget = ')
% for i = 1:size(targetVector)
%     disp([ num2str(targetVector(i))  '   '])
% end
for i = 1:size(targetVector)
    cost = cost + ( targetVector(i) - predicted(i) )^2;
end
% disp(['cost = '  num2str(cost)])
% cost = -log( cost ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Backprop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [tree.nodeVecDeltas,tree.NN_deltas,paddingDelta] = backpropPool(tree, label, params);
[tree.nodeVecDeltas,paddingDelta] = backpropPool(tree, targetVector, params);

deltaDown_vec = zeros(params.wordSize,1);
deltaDown_op = zeros(params.wordSize,params.wordSize);

topNode = tree.getTopNode();
% [df_s_Wv,df_s_Wo,df_s_W,df_s_WO] = backpropAll(tree,W,WO,Wo,params,deltaDown_vec,deltaDown_op,topNode,size(Wv,2),indicies,sNN);
[df_s_Wv,df_s_Wo,df_s_W,df_s_WO] = backpropAll(tree,W,WO,Wo,params,deltaDown_vec,deltaDown_op,topNode,size(Wv,2));

%Backprop into Padding
df_s_Wv(:,1) = df_s_Wv(:,1) + paddingDelta; % updating the word-vectors 