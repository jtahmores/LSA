
clear all;
clc
% Set algorithm parameters
options.ker = 'linear';     % 'primal' | 'linear' | 'rbf' % Type of kernel
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10; % Number of iterations
meanLSA=[];
srcStr = {'Caltech10','Caltech10','Caltech10','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStr = {'amazon','webcam','dslr','Caltech10','webcam','dslr','Caltech10','amazon','dslr','Caltech10','amazon','webcam'};
options.k=100;
options.lambda = 0.5;
options.knn=10;
options.sigma = 0.3;
options.theta = 0.1;

for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    fprintf('%s\n',options.data);
    load(['datasets/office-caltech-surf/' src '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xs = zscore(fts,1);
    Xs = Xs';
    Ys = labels;
    load(['datasets/office-caltech-surf/' tgt '_SURF_L10.mat']);
    fts = fts ./ repmat(sum(fts,2),1,size(fts,2));
    Xt = zscore(fts,1);
    Xt = Xt';
    Yt = labels;
    Cls = [];
    
    for t=1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = LSA(Xs,Xt,Ys,Cls,options); % A: Adaptation matrix, Z: projected data
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt);
        fprintf('LSA+NN=%0.2f\n',acc*100);
    end
    
    fprintf(fid,'LSA accuracies in iterations::\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc);
    meanLSA=[meanLSA;acc*10];
    Acc=[];
    Cls=[];
end
fprintf('mean of LSA: %0.2f\n',mean(meanLSA));
