
clear all;
clc
options.ker = 'rbf';     % 'primal' | 'linear' | 'rbf' % Type of kernel
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10; % Number of iterations
meanLSA=[];
Acc=[];
srcStrDecaf12 = {'caltech','caltech','caltech','amazon','amazon','amazon','webcam','webcam','webcam','dslr','dslr','dslr'};
tgtStrDecaf12 = {'amazon','webcam','dslr','caltech','webcam','dslr','caltech','amazon','dslr','caltech','amazon','webcam'};
options.k= 40;
options.lambda = 0.0005;
options.sigma =  0.2;
options.theta =  0.1;
options.knn=10;

for iData = 1:12
    src = char(srcStrDecaf12{iData});
    tgt = char(tgtStrDecaf12{iData});
    options.data = strcat(src,'-vs-',tgt);
    fprintf('Data=%s \n',options.data);
    src_data= load(strcat('datasets/Office-Caltech-decaf6/',src));
    Xs = src_data.feas;
    Ys = src_data.labels;
    Xs = normr(Xs);
    Xs = Xs';
    tgt_data= load(strcat('datasets/Office-Caltech-decaf6/',tgt));
    Xt = tgt_data.feas;
    Yt = tgt_data.labels;
    Xt = normr(Xt);
    Xt = Xt';
    Cls = [];
    
    for t=1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = LSA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt);
        fprintf('LSA+NN=%0.2f\n',acc*100);
        Acc=[Acc;acc*100];
    end
    fprintf('LSA:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc2);
    meanLSA=[meanLSA;Acc(10)];
end
fprintf('mean of LSA: %0.2f\n',mean(meanLSA));
