function main()
save proj3.mat;
load data.mat;

N = size(training_set,1);
features = size(training_set,2);
num_classes = 10;

T = zeros(N, num_classes);
for i = 1:num_classes
    T(:, i) = (training_label==i-1);  
end

W = training_LR(training_set, T, test_set, test_label);
Wlr = W(1:features,:);
blr = ones(1,num_classes);

save('proj3.mat','Wlr','-append');
save('proj3.mat','blr','-append');
%%%%%Neural%%%%%%%
wji = rand(785,50)/1000;
wkj = rand(51, 10)/1000;
Enew=1;
Eold = 2;
EE = [];
eta = 0.001;
while(abs(Enew-Eold) > 0.01),
   for i=1:60000,
       Eold = Enew;
       X = trainX(i,:);
       z = tanh(X*wji);
       z = [1 z];
       a = z*wkj;
       y = zeros(1,10);
       for j = 1:10,
           y(j) = exp(a(j))./sum(exp(a));
       end
       Ecross = -sum(trainT(i,:).*log(y+0.001));
       EE = [EE Ecross];
       plot(EE)
       drawnow;
       delk = y - trainT(i,:);
       zz = z(1,2:size(z,2));
       delj = (1-tanh(zz).^2)*sum(wkj*delk');
       delEkj = z'*delk;
       delEji = X'*delj;
       wji = wji - eta*delEji;
       wkj = wkj - eta*delEkj;
   end
end
Wnn1 = wji(2:size(wji,1),:);
bnn1 = wji(1,:);
Wnn2 = wkj(2:size(wkj,1),:);
bnn2 = wkj(1,:);
h='tanh';

save('proj3.mat','Wnn1','-append');
save('proj3.mat','Wnn2','-append');
save('proj3.mat','bnn1','-append');
save('proj3.mat','bnn2','-append');
save('proj3.mat','h','-append');

end






function ws = training_LR( x, T, xv, Lv)
batchsize = 100;
max_passes = 400;
learn_rate.a = 1; % parameters for the learning rate of stochastic gradient descent
learn_rate.b = 1; 
 
M = size(x,2)+1; 
K = size(T,2); 
 
 % Add a bias term to the features (i.e. a feature of all 1's)
phi = [x, ones(size(x,1),1)];
phiV = [xv, ones(size(xv,1),1)];
 
w0 = randn((M-1),K)*0.1;  % initial weights set randomly
w0 = [w0;ones(1,10)]; %add bias
ws = SGD(phi, T, phiV, Lv, w0, batchsize, max_passes, learn_rate);
end
 
 
function w = SGD(X, T, validationset ,validationlabels, w0, batchsize, maxpasses, learn_rate)
    N = size(X,1);
    w = w0;
    epoch = 0;           
    stepsize = learn_rate.a / (learn_rate.b+epoch);
    batchstart = 1;
    
    while(epoch < maxpasses)
        batchend   = min(N,batchstart+batchsize-1);
        dw = Grad_err(w,X(batchstart:batchend,:),T(batchstart:batchend,:));
        w = w - stepsize*dw;
        batchstart = batchstart+batchsize;
        if(batchstart>N)
            % tests the classification on the validation set and report
            % results.  Could use this to inform learning rate (stepsize)
            errorrate = LR_err(w, validationset, validationlabels);
            batchstart = 1;      % start over with the data
            epoch = epoch+1;
            stepsize = learn_rate.a / (learn_rate.b+epoch);
        end
    end
end
 
 
function gradE = Grad_err(ws,phi,T)
    N = size(phi,1); 
    K = size(T,2);  
 
    gradE = zeros(size(ws));
    for n = 1:N
        pn = exp(ws'*phi(n,:)'); %'
        yn = pn/sum(pn);            
        for i=1:K
            gradE(:,i) = gradE(:,i) + (yn(i) - T(n,i))*phi(n,:)'; %'
        end
    end
    gradE = (1/N)*gradE;  % normalize by the number of datapoints
end
 
function err_rt = LR_err( ws, testphis, testlabels )
misclassified =0;
target = size(testphis,1);
    for i=1:target
        as = ws'*testphis(i,:)'; %'
        [~, label(i)] = max(as);
        label(i) = label(i) - 1;
        if(label (i) ~= testlabels (i))
            misclassified = misclassified+1;
        end
    end
    err_rt = misclassified / size(testphis,1);
end



