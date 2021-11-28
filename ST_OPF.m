function [L,t]=ST_OPF(label_x,label_x_t,unlabel_x)
%% initialize variables
L=label_x; % labeled samples
U=unlabel_x;% unlabeled samples
t=label_x_t;%  the class label of labeled samples
count=1;% iterative count
data=[L;U];  % semi-supervised data sets
L_index=[1:1:size(L,1)]';  % the index of labeled samples
U_index=[size(L,1)+1:1:size(data,1)];  % the index of unlabeled samples
%% construct an optimum path forest (OPF) on semi-supervised data sets 
P=OPF(data,L);
% arrows implies the relationship between L and U
arrows=P;
%% Self-training process in STOPF
count=1;
while 1 
    index=[];
    for i=1:length(L_index)
        pos=find(arrows(U_index)==L_index(i));
        index=[index;U_index(pos)'];
    end
    if length(index)==0
        break;
    end
    classifyU=data(index,:);
    KNN_index=KNNC(L,t,classifyU,3);
    %%
    U_index=setdiff(U_index,index);
    for i=1:length(index)
        L_index=[L_index;index(i)];
        t=[t;KNN_index(i)];
    end
    L=data(L_index,:);
    U=data(U_index,:);
    count=count+1;
    if size(U,1)==0
        break;
    end
end
end