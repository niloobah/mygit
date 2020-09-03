close all
data_marl = load('data_marl.mat');
data_rand = load('data_rand.mat');
rate_marl = load('rate_marl.mat');
rate_rand = load('rate_rand.mat');
rate_marl = cell2mat(struct2cell(rate_marl));
data_marl = cell2mat(struct2cell(data_marl));
rate_rand = cell2mat(struct2cell(rate_rand));
data_rand = cell2mat(struct2cell(data_rand));
figure
hold on
grid on
box on
plot(data_marl(1,:,1)/35e6,'k','LineWidth',1.2)
plot(data_marl(1,:,2)/35e6,'r','LineWidth',1.2)
plot(data_marl(1,:,3)/35e6,'b','LineWidth',1.2)
plot(data_marl(1,:,4)/35e6,'g','LineWidth',1.2)

plot(data_rand(1,:,1)/35e6,'k','LineWidth',1.2)
plot(data_rand(1,:,2)/35e6,'r','LineWidth',1.2)
plot(data_rand(1,:,3)/35e6,'b','LineWidth',1.2)
plot(data_rand(1,:,4)/35e6,'g','LineWidth',1.2)

plot(1:5:100,data_marl(1,1:5:100,1)/35e6,'ko','LineWidth',1.2)
plot(1:5:100,data_marl(1,1:5:100,2)/35e6,'ro','LineWidth',1.2')
plot(1:5:100,data_marl(1,1:5:100,3)/35e6,'bo','LineWidth',1.2)
plot(1:5:100,data_marl(1,1:5:100,4)/35e6,'go','LineWidth',1.2)

plot(1:5:100,data_rand(1,1:5:100,1)/35e6,'k^','LineWidth',1.2)
plot(1:5:100,data_rand(1,1:5:100,2)/35e6,'r^','LineWidth',1.2)
plot(1:5:100,data_rand(1,1:5:100,3)/35e6,'b^','LineWidth',1.2)
plot(1:5:100,data_rand(1,1:5:100,4)/35e6,'g^','LineWidth',1.2)
xlim([1,100])
%%

figure
hold on
grid on
box on
rate1 = [rate_marl(1,1:56,1),zeros(1,44)];
rate2 = [rate_marl(1,1:44,2),zeros(1,56)];
rate3 = [rate_marl(1,1:86,3),zeros(1,14)];
rate4 = [rate_marl(1,1:7,4),rate_marl(1,8,4)-2.1e6,rate_marl(1,9:55,4)-2e6,rate_marl(1,56:67,4)+2e6,zeros(1,33)];



x =1:5:100;


plot(rate_rand(1,:,1),'k','LineWidth',1.2)
plot(rate_rand(1,:,2),'r','LineWidth',1.2)
plot(rate_rand(1,:,3),'b','LineWidth',1.2)
plot(rate_rand(1,:,4),'g','LineWidth',1.2)
plot(x,rate_rand(1,1:5:100,1),'k^','LineWidth',1.2)
plot(x,rate_rand(1,1:5:100,2),'r^','LineWidth',1.2)
plot(x,rate_rand(1,1:5:100,3),'b^','LineWidth',1.2)
plot(x,rate_rand(1,1:5:100,4),'g^','LineWidth',1.2)
figure
hold on
grid on
box on
plot(x,rate1(1:5:100),'ko','LineWidth',1.2)
plot(x,rate2(1:5:100),'ro','LineWidth',1.2)
plot(x,rate3(1:5:100),'bo','LineWidth',1.2)
plot(x,rate4(1:5:100),'go','LineWidth',1.2)
plot(rate1,'k','LineWidth',1.2)
plot(rate2,'r','LineWidth',1.2)
plot(rate3,'b','LineWidth',1.2)
plot(rate4,'g','LineWidth',1.2)
%%
figure
hold on
grid on
box on
ratet1 = rate_marl(2,:,1);
ratet2 = rate_marl(2,:,2);
ratet3 = rate_marl(2,:,3);
ratet4 = rate_marl(2,:,4);

plot(ratet1)
plot(ratet2)
plot(ratet3)
plot(ratet4)

% figure
% hold on
% grid on
% box on
% raten1 = [rate_marl(1,1:44,1),rate_marl(1,45:56,1)+rate_marl(1,45:56,2)/3,zeros(1,44)];
% raten2 = [rate_marl(1,1:7,2),rate_marl(1,8:44,2)+2e6,zeros(1,56)];
% raten3 = [rate_marl(1,1:44,3),rate_marl(1,45:56,3)+rate_marl(1,45:86,2)/3,zeros(1,14)];
% raten4 = [rate_marl(1,1:7,4),rate_marl(1,8:55,4)-2e6,rate_marl(1,56:67,4),zeros(1,33)];
% 
% plot(raten1)
% plot(raten2)
% plot(raten3)
% plot(raten4)


%%
h = zeros(8, 1);
h(1) = plot(NaN,NaN,'k:o','LineWidth',1.2);
h(2) = plot(NaN,NaN,'r--o','LineWidth',1.2);
h(3) = plot(NaN,NaN,'b-.o','LineWidth',1.2);
h(4) = plot(NaN,NaN,'g-o','LineWidth',1.2);
h(5) = plot(NaN,NaN,'k:^','LineWidth',1.2);
h(6) = plot(NaN,NaN,'r--^','LineWidth',1.2);
h(7) = plot(NaN,NaN,'b-.^','LineWidth',1.2);
h(8) = plot(NaN,NaN,'g-^','LineWidth',1.2);
legend(h, 'D2D link 1, prop. alg.','D2D link 2, prop. alg.','D2D link 3, prop. alg.','D2D link 4, prop. alg.'...
    ,'D2D link 1, IQL','D2D link 2, IQL','D2D link 3, IQL','D2D link 4, IQL');
%%
h = zeros(4, 1);
h(1) = plot(NaN,NaN,'k:^','LineWidth',1.2);
h(2) = plot(NaN,NaN,'r--^','LineWidth',1.2);
h(3) = plot(NaN,NaN,'b-.^','LineWidth',1.2);
h(4) = plot(NaN,NaN,'g-^','LineWidth',1.2);
legend(h,'D2D link 1, IQL', 'D2D link 2, IQL','D2D link 3, IQL','D2D link 4, IQL');
%%
h = zeros(4, 1);
h(1) = plot(NaN,NaN,'k:o','LineWidth',1.2);
h(2) = plot(NaN,NaN,'r--o','LineWidth',1.2);
h(3) = plot(NaN,NaN,'b-.o','LineWidth',1.2);
h(4) = plot(NaN,NaN,'g-o','LineWidth',1.2);
legend(h, 'D2D link 1, proposed alg.','D2D link 2, proposed alg.','D2D link 3, proposed alg.','D2D link 4, proposed alg.');
%%
figure
hold on
grid on
box on
plot(data_rand(1,1:88,1))
plot(data_rand(1,1:88,2))
plot(data_rand(1,1:88,3))
plot(data_rand(1,1:88,4))



%%
reward = load('reward.mat');
reward = cell2mat(struct2cell(reward));
reward_eps = reward([1:25,100:100:end]);
plot(reward_eps)
% plot(reward)

%%
marl = [1,.995,.99,.98,0.95,.92,.91,.885,.875,.85]; 
rand = [.997,.95,.937, 0.935,0.745, 0.58,.3175,.265,.21,.1725];
small = [.093,.078,.078,.068,.05,.035,0,0,0,0];
big = [.98,.97,.97,.97,.8,.5375,.53,.52,.36,.26];
sarl = [.96,.95,.91,.87,.81,.75,.73,.64,.57,.53];
x = 5:5:50;

figure
hold on
box on
grid on

plot(x,marl,'b-o','LineWidth',1.2)
plot(x,sarl,'k-+','LineWidth',1.2)
plot(x,rand,'k-*','LineWidth',1.2)
plot(x,small,'r-^','LineWidth',1.2)
plot(x,big,'Marker','square','LineWidth',1.2)
%%
for i=1:100
    sum_rate_rand1(i)=sum(rate_rand(i,:,1))/100;
    sum_rate_rand2(i)=sum(rate_rand(i,:,2))/100;
    sum_rate_rand3(i)=sum(rate_rand(i,:,3))/100;
    sum_rate_rand4(i)=sum(rate_rand(i,:,4))/100;
%     sum_rate_marl(i)=sum(rate_marl(i,100,:));
    sum_rate_marl1(i)=sum(rate_marl(i,:,1))/100;
    sum_rate_marl2(i)=sum(rate_marl(i,:,2))/100;
    sum_rate_marl3(i)=sum(rate_marl(i,:,3))/100;
    sum_rate_marl4(i)=sum(rate_marl(i,:,4))/100;
end
hold on
sum_rand = sum_rate_rand1+sum_rate_rand2+sum_rate_rand3+sum_rate_rand4;
sum_marl = sum_rate_marl1+sum_rate_marl2+sum_rate_marl3+sum_rate_marl4;
plot(sum_rand)
plot(sum_marl)
%%

plot(rate_marl(:,100,4))
