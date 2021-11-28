function P=OPF(data,label_x)
%% optimum-path forest based
%% initialize nodes
P=zeros(size(data,1),1);
C=zeros(size(data,1),1);
for i=size(label_x,1)+1:size(data,1)
    C(i)=inf;
end
queue=[];
for i=1:size(label_x,1)
    C(i)=0;
    queue=[queue;i];
end
%%
count=1;
while 1
    %% 
    queue_C_index=queue;
    % 
   [value,index]= min(C(queue_C_index));
    idx=queue(index);
    if length(idx)==0
        break;
    end
    % remove a sample from queue and the sample has the min C
    queue(index)=[];
    count=count+1;
    KNN_idx=knnsearch(data,data(idx,:),'NSMethod','kdtree','K',size(data,1));
    KNN_idx(1)=[];
    for i=1:length(KNN_idx)
        if C(KNN_idx(i))>C(idx)  % for each x  with c(x)>c(idx)
            data1=data(idx,:);
            data2=data(KNN_idx(i),:);
            cst=max(C(idx),pdist2( data1, data2) );
            if cst<C(KNN_idx(i))
                % 
                if C(KNN_idx(i))~=inf
                    pos=find(queue==KNN_idx(i));
                    queue(pos)=[];
                    
                end
                % sample  "KNN_idx(i)" connects to "idx", while updating C and queue
                P(KNN_idx(i))=idx;
                C(KNN_idx(i))=cst;
                queue=[queue;KNN_idx(i)];
            end
        end
    end
    queue=unique(queue);
end
end

