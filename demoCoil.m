
clear all;
clc
% Set algorithm parameters
options.ker = 'linear';     % 'primal' | 'linear' | 'rbf' % Type of kernel
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10; % Number of iterations
meanLSA=[];
Acc=[];
options.k=20;
options.lambda = 0.0005;
options.sigma =0.8;
options.theta =  0.001;
options.knn=10;

for dataStr ={'COIL_1','COIL_2'};
    data = strcat(char(dataStr));
    options.data = data;
    load(strcat('datasets/COIL20/',data));
    X_src = X_src*diag(sparse(1./sqrt(sum(X_src.^2))));
    X_tar = X_tar*diag(sparse(1./sqrt(sum(X_tar.^2))));
    Xs=X_src;
    Xt=X_tar;
    Ys=Y_src;
    Yt=Y_tar;
    X=[Xs';Xt'];
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
        Acc = [Acc;acc*100];
    end
    fprintf('LSA accuracies in iterations:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc);
    meanLSA=[meanLSA;Acc(10)];
    Acc=[];
    Cls=[];
    
end
fprintf('mean of LSA: %0.2f\n',mean(meanLSA));
