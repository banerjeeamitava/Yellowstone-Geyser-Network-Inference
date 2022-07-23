clear

compare=zeros(18,12,10,10);

%1 month=43200 min;
delays=[5,10,30,60,120,240,480,960,1440,2880,7200,14400]/5;%in min (5 min sampling), specify set of delays
dl_labels=["5 min","10 min", "30 min", "1 hr", "2 hr", "4 hr", "8 hr", "16 hr", "1 day", "2 day", "5 day", "10 day"];
delays=floor(delays);
% rng shuffle

%%read data (time-series)
T=readtable('collation_revised.csv','NumHeaderLines',1);%You will need to download the original data from https://doi.org/10.5061/dryad.n5tb2rbrm separately.

x=zeros(833844,10);

x(:,1)=table2array(T(:,4));
x(:,2)=table2array(T(:,5));
x(:,3)=table2array(T(:,6));
x(:,4)=table2array(T(:,7));
x(:,5)=table2array(T(:,8));
x(:,6)=table2array(T(:,10));
x(:,7)=table2array(T(:,11));
x(:,8)=table2array(T(:,12));
x(:,9)=table2array(T(:,13));
x(:,10)=table2array(T(:,15));

x=x';

names=["BE"; "CA"; "DE"; "DO"; "GR"; "LN"; "LS"; "OF"; "PT"; "PM"]';%Geyser names
figure

for imonth=1:1
for idel=6:6
   disp([imonth,idel])
    
idelay=delays(1,idel);%set delay time step

xplot=x(:,(imonth-1)*43200+1:5:imonth*43200);%5 min sampling for one month
data=xplot(:,1:5700);%input to reservoir computer
diff_data=xplot(:,1+idelay:5700+idelay);%delayed prediction

% xplot=x(:,1:5:end);
% data=xplot(:,1:160000);
% diff_data=xplot(:,1+idelay:160000+idelay);

%% remove NaNs
h=isnan(data);
h=sum(h,1);
h(h>0)=1;
f=find(h==1);%find location of NaNs in data
data(:,f)=[];%delete NaNs from data
diff_data(:,f)=[];%delete corresponding elements from diff_data

h=isnan(diff_data);%find additional NaNs in diff_data
h=sum(h,1);
h(h>0)=1;
f=find(h==1);%find location of NaNs in diff_data
data(:,f)=[];%delete NaNs from data
diff_data(:,f)=[];%delete corresponding elements from diff_data
%% Data for Reservoir training
measurements1 = data(:,1:end);% + z;
measurements2 = diff_data(:, 1:end);


%% Nelder-Mead hyperparameter optimization

p=[6.64,0.06];%best results from N-M of 30000 steps at idelay=1 and kept fixed, RC kept fixed as well
rho=0.9;%fixed 


% fun = @(p)residual(p,measurements1,measurements2,rho);
% p0 = p;%Starting values of Spectral Radius, Av. Degree, Input Scaling, beta
% options = optimset('PlotFcns',@optimplotx,'Display','iter','TolX',1e-1,'TolFun',1e-8);
% p = fminsearch(fun,p0,options)


resparams=struct();
%% train reservoir
[num_inputs,~] = size(measurements1);
resparams.radius = rho; % spectral radius
resparams.degree = p(1,1); % connection degree
approx_res_size = 3000; % reservoir size
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = p(1,2); % input weight scaling
resparams.bias=0;%bias
resparams.leakage=1;
resparams.train_length = size(data,2)-10; % number of points used to train
resparams.num_inputs = num_inputs; 
resparams.predict_length = 2000; % number of predictions after training
resparams.predict_length_max=resparams.predict_length;
resparams.beta = 0.0001; %regularization parameter

%% Reservoir dynamics
%A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);
%q = resparams.N/resparams.num_inputs;
%win = zeros(resparams.N, resparams.num_inputs);
%for i=1:resparams.num_inputs
%      rng(i+50,'twister')
%     ip = resparams.sigma*(-1 + 2*rand(q,1));
%     win((i-1)*q+1:i*q,i) = ip;
% end
 %states = reservoir_layer(A, win, data, resparams);



[xx, w_out, A, win,r] = train_reservoir(resparams, measurements1,measurements2);%Train and save w_out matrix for future use

%% Predicting only one component from itself with same w_out (uncomment and run this part if you wish to evaluate prediction from one geyser only)
%predict_distance=zeros(2,10);
%for igeyser=1:10
    %disp(igeyser)
%m=zeros(size(measurements1));
%m(igeyser,:)=measurements1(igeyser,:);
%r1 = reservoir_layer(A, win, m, resparams);
%d=w_out*r;%original training fit
%d1=w_out*r1;%only using one geyser
%d0=measurements2(:,1:5689);
%predict_distance(1,igeyser)=sum(abs(d0(igeyser,:)-d(igeyser,:)).^2,'all');
%predict_distance(2,igeyser)=sum(abs(d0(igeyser,:)-d1(igeyser,:)).^2,'all');
%end
%save("predict_distance.mat","predict_distance")




 disp(['training Done'])
% [output,r] = predict(A,win,resparams,xx,data(:,resparams.train_length:resparams.train_length+resparams.predict_length-1),w_out);%Prediction for the Training Time Series Data


%% Jacobian Estimation
av_length=floor(resparams.train_length*0.8);

%% If predicting x(t+tau)from x(t)
 conn1=zeros(size(win,2));
 B=A+win*w_out;
 for it=resparams.train_length-av_length:resparams.train_length
     
     if mod(it,1000)==0
     disp(it)
     end
     
     xx=r(:,it);
     A2=B*xx;
         
     mat1=zeros(size(w_out));
 
     for i1=1:resparams.N
         mat1(:,i1)=w_out(:,i1)*(sech(A2(i1)))^2;
     end
     
            conn1=conn1+abs(mat1*(win+A*pinv(w_out)));
% %           conn1=conn1+(mat1*(win+A*pinv(w_out)));
 
 end
 conn1=conn1/(av_length);
 
 compare(imonth,idel,:,:)=conn1(:,:);

figure
for idel1=1:12
subplot(3,4,idel1)
conn1=zeros(10);
conn1(:,:)=compare(1,idel1,:,:);
imagesc(conn1-diag(diag(conn1)))
pbaspect([1 1 1])
xticks(1:1:10)
yticks(1:1:10)
xticklabels(names)
yticklabels(names)
xtickangle(45)
ytickangle(45)
title(dl_labels(1,idel1))
colorbar
set(gca,'FontSize',10)
end

end
end

%% maximize strength over delays
comparemax=zeros(10);
dls=zeros(10);
f=zeros(1,12);
for i=1:10
    for j=1:10
        if (i~=j)
            f(1,:)=compare(1,:,i,j);
            [M,I]=max(f,[],'linear');
            comparemax(i,j)=M;
            dls(i,j)=I*5;
        end
    end
end
%% Plotting monthly data
% % for i=13:18
%     figure
%     for j=1:12
%         subplot(3,4,j)
%         gg=zeros(10);
% %         gg(:,:)=compare(i,j,:,:);
%          gg(:,:)=compare1(j,:,:);
% 
%         imagesc(gg-diag(diag(gg)))
%         pbaspect([1 1 1])
%         xticks(1:1:10)
%         yticks(1:1:10)
%         xticklabels(names)
%         yticklabels(names)
%         xtickangle(45)
%         ytickangle(45)
%         title(dl_labels(1,j))
%         colorbar
%         set(gca,'FontSize',10)
% 
%         
%     end
% %     sgtitle([num2str(i),'-th month'])
% %     set(gca,'FontSize',10)
% % end
% % 


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, wout, A, win,states] = train_reservoir(resparams, data1,data2)

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);
q = resparams.N/resparams.num_inputs;
win = zeros(resparams.N, resparams.num_inputs);
for i=1:resparams.num_inputs
     rng(i+50,'twister')
    ip = resparams.sigma*(-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end
states = reservoir_layer(A, win, data1, resparams);
wout = train(resparams, states, data2(:,1:resparams.train_length));
x = states(:,end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = generate_reservoir(size, radius, degree)
  rng(1,'twister');
sparsity = degree/size;
while 1
A = sprand(size, size, sparsity);
e = max(abs(eigs(A)));

if (isnan(e)==0)%Avoid NaN in the largest eigenvalue, in case convergence issues arise
    break;
end

end
A = (A./e).*radius;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function states = reservoir_layer(A, win, input, resparams)

states = zeros(resparams.N, resparams.train_length);
for i = 1:resparams.train_length-1
    states(:,i+1) = (1-resparams.leakage)*states(:,i)+resparams.leakage*tanh(A*states(:,i) + win*input(:,i));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w_out = train(params, states, data)

%Take some time points out of training if you want
% T=size(states,2);
% states(:,floor(T/2)-20:floor(T/2)+20)=[];
% data(:,floor(T/2)-20:floor(T/2)+20)=[];



beta = params.beta;
rng(2,'twister');
idenmat = beta*speye(params.N);
% states(2:2:params.N,:) = states(2:2:params.N,:).^2;
w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=residual(x,data1,data2,rho)
measured_vars = 1:1:size(data1,1);
num_measured = length(measured_vars);
measurements1 = data1(measured_vars, :);% + z;
measurements2 = data2(measured_vars, :);% + z;

resparams=struct();
%train reservoir
[num_inputs,~] = size(measurements1);
resparams.radius = rho; % spectral radius, around 0.9
resparams.degree = x(1); % connection degree, around 3
approx_res_size = 3000; % reservoir size
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = x(2); % input weight scaling, around 0.1
resparams.leakage=1;
resparams.train_length = 30000; % number of points used to train
resparams.num_inputs = num_inputs; 
resparams.predict_length = 1000; % number of predictions after training
resparams.beta = 0.0001; %regularization parameter, near 0.0001

[~, H, ~, ~,r] = train_reservoir(resparams, measurements1,measurements2);
output1=H*r;
f=sum((output1(:,floor(resparams.train_length*0.1):resparams.train_length)-data2(:,floor(resparams.train_length*0.1):resparams.train_length)).^2,'all');
end

