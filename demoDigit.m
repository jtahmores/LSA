
clear all;
clc
options.ker = 'rbf';
options.gamma = 1.0;
T = 10;
Acc=[];
meanLSA=[];
options.k=220 ;
options.lambda = 0.00001;
options.sigma =0.1;
options.theta =  0.00005;
options.knn=10;

for dataStr = {'USPS_vs_MNIST','MNIST_vs_USPS'}
    
    % Preprocess data using L2-norm
    data = strcat(char(dataStr));
    options.data = data;
    load(strcat('datasets/digit/',data));
    X_src = X_src*diag(sparse(1./sqrt(sum(X_src.^2))));
    X_tar = X_tar*diag(sparse(1./sqrt(sum(X_tar.^2))));
    Xs=X_src;
    Xt=X_tar;
    Ys=Y_src;
    Yt=Y_tar;
    Cls = [];
    
    for t=1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = LSA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        Cls = knnclassify(Zt',Zs',Ys,1);
        acc = length(find(Cls==Yt))/length(Yt);
        fprintf('JDA+NN=%0.2f\n',acc*100);
        fprintf(fid,'JDA+NN=%0.2f\n',acc*100);
        Acc = [Acc;acc*100];
        
        
    end
    fprintf('LSA:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',Acc);
    meanLSA=[meanLSA;Acc(10)];
    Acc=[];
    Cls=[];
       
end
fprintf('mean of LSA: %0.2f\n',mean(meanLSA));



