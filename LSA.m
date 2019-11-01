function [Z,A] = LSA(Xs,Xt,Ys,Yt0,options)

% Joint Local and Statistical Discriminant Learning via Feature Alignment.
% Signal, Image and Video Processing. https://doi.org/10.1007/s11760-019-01587-1


% Contact:  Jafar Tahmoresnezhad (tahmores@gmail.com)
%           Elahe Gholenji (elahegholenji@gmail.com)

k = options.k;
lambda = options.lambda;
ker = options.ker;
gamma = options.gamma;
sigma=options.sigma;
theta=options.theta;
data = options.data;


% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

% Construct MMD matrix
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e*e'*C;
if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e';
    end
end
M = M/norm(M,'fro');

% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);
manifold.k = options.knn;

% Construct Laplacian matrix
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
W = graphLaplacian(X',manifold);
Dw=zeros(n,n);

Lb=zeros(n,n);
Ww=zeros(n,n);
if ~isempty(Yt0)
    Cls=Yt0;
    nm=ns+nt;
    Y_1=[Ys;Cls];
    Label = unique(Y_1);
    nLabel = length(Label);
    Ww = zeros(nm,nm);
    Wb = ones(nm,nm) ;
    for idx=1:nLabel
        classIdx = find(Y_1==Label(idx));
        Ww(classIdx,classIdx) = 1;
        Wb(classIdx,classIdx) = 0;
    end
    Ww=Ww.*W;
    Wb=Wb.*W;
    Dw = diag(sparse(sqrt(1./sum(Ww))));
    Lw = speye(nm)-Dw*Ww*Dw;
    
    Db = diag(sparse(sqrt(1./sum(Wb))));
    Lb = speye(nm)-Db*Wb*Db;
    
end


%MMD-discriminative matrix
%M3=>zero matrix (size:n)
%M2=>source to source
%M4=>target to target
%M5=>source to target
%M6=>target to source

%M S->S
M2=zeros(ns,ns);
M3=zeros(ns,ns);

for c = reshape(unique(Ys),1,C)
    e = zeros(ns,1);
    
    e(Ys==c) = 1/length(find(Ys==c));
    for cc =reshape(unique(Ys),1,C)
        if cc==c
            continue;
        else
            e(Ys==cc) = -1/length(find(Ys==cc));
        end
        M3 = M3 + e*e';
        
    end
end


M6=zeros(n,n);
M5=zeros(n,n);
M4=zeros(nt,nt);

if ~isempty(Yt0)
    %M T->T
    M4=zeros(nt,nt);
    
    for c1 = reshape(unique(Ys),1,C)
        e = zeros(nt,1);
        
        e(Yt0==c1) = 1/length(find(Yt0==c1));
        for cc1 =reshape(unique(Ys),1,C)
            if cc1==c1
                continue;
            else
                e(Yt0==cc1) = -1/length(find(Yt0==cc1));
                
            end
            M4 = M4 + e*e';
            
        end
        
    end
    
    %M S->T
    M5=zeros(n,n);
    
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        
        e(Ys==c) = 1/length(find(Ys==c));
        for cc =reshape(unique(Ys),1,C)
            if cc==c
                continue;
            else
                e(Yt0==cc) = -1/length(find(Yt0==cc));
                
            end
            M5 = M5 + e*e';
        end
        
    end
    
    
    %M T->S
    
    
    
    
    for c1 = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        
        e(Ys==c1) = 1/length(find(Ys==c1));
        for cc1 =reshape(unique(Ys),1,C)
            if cc1==c1
                %cc1=cc1+1;
                continue;
            else
                e(Yt0==cc1) = -1/length(find(Yt0==cc1));
            end
            M6 = M6 + e*e';
            
        end
        
    end
    
end
MM=[M3,zeros(ns,nt);zeros(nt,ns),M4];


%within-class matrix
Qs = ones(ns,m);
Qt = zeros(nt,m);
XXs = Xs';
XXt = Xt';
for c = reshape(unique(Ys),1,C)
    Qs(Ys==c,:) = XXs(Ys==c,:) - repmat(mean(XXs(Ys==c,:)),length(find(Ys==c)),1);
    if ~isempty(Yt0)
        Qt(Yt0==c,:) = XXt(Yt0==c,:) - repmat(mean(XXt(Yt0==c,:)),length(find(Yt0==c)),1);
    end
end
Q = [Qs; Qt];

K = kernel(ker,X,[],gamma);
[A,~] = eigs(K*(M+theta*Dw)*K'+Q*Q'+lambda*eye(n),K*(H+sigma*Lb+(1-sigma)*Ww+(MM+M5+M6))*K',k,'SM');
Z = A'*K;


end
