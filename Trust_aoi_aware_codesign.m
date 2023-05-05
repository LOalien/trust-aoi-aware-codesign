%%初始化
clc;
%参数设置
MAXTIME=100;%在线仿真100个时刻
LINKNUM=4;%链路数目
lamda=0.35;%
%总共有m条link
J_inf_max=99999*ones(MAXTIME,LINKNUM);
Eage=9999*ones(MAXTIME,LINKNUM);% 瞬时age 
trust=zeros(1,LINKNUM);
linkIndex=zeros(MAXTIME,1);
minJ=zeros(MAXTIME,1);
minAgeJ=zeros(MAXTIME,1);
randomJ=zeros(MAXTIME,1);
aoi=zeros(LINKNUM,1);
input_data1=readtable('./TensorFlow/onlinedataset_sensor1.csv');
input_data2=readtable('./TensorFlow/onlinedataset_sensor2.csv');
input_data3=readtable('./TensorFlow/onlinedataset_sensor3.csv');
input_data4=readtable('./TensorFlow/onlinedataset_sensor4.csv');
% 将表格转换为矩阵
input_data1 = table2array(input_data1);
input_data2 = table2array(input_data2);
input_data3 = table2array(input_data3);
input_data4 = table2array(input_data4);
%去掉每个数据集的第一行和第一列
input_data1(1, :) = [];
input_data1(:, 1) = [];
input_data2(1, :) = [];
input_data2(:, 1) = [];
input_data3(1, :) = [];
input_data3(:, 1) = [];
input_data4(1, :) = [];
input_data4(:, 1) = [];
%% 控制参数求解
%初始化
Wk = eye(2);
Uk = 1;
%噪声协方差
Q = [0.1 0;0 0.1];
R = 1;

A=[0.37 0;0.67 1];
B=[0.67 0.37]';
[K,S_inf,e]=dlqr(A,B,Wk,Uk);

%% online processing
for time=1:MAXTIME
% 模拟随机测量得到的AoI
Eage(time,:)=10*rand(LINKNUM,1);

% 求control cost 
% 当前的trust和age，
sum_cost = zeros(LINKNUM,1);
for j=1:LINKNUM 
	aoi(j)=floor(Eage(time,j));
    % 记录最小化age的索引
    [~,tmpindex]=min(aoi);
    for i=1:aoi(j)
         sum_cost(j)=sum_cost(j)+trace(S_inf*A^i*Q*(A^i)');
    end
end
% 调用python代码里训练好的DNN模型，预测每个sensor channel pair的trust
% 构造要调用的命令，包括传递参数
cmd1 = sprintf('conda activate GANbp && python ./TensorFlow/trust_predictor.py %f %f %f %f', input_data1(time,:));
cmd2 = sprintf('conda activate GANbp && python ./TensorFlow/trust_predictor.py %f %f %f %f', input_data2(time,:));
cmd3 = sprintf('conda activate GANbp && python ./TensorFlow/trust_predictor.py %f %f %f %f', input_data3(time,:));
cmd4 = sprintf('conda activate GANbp && python ./TensorFlow/trust_predictor.py %f %f %f %f', input_data4(time,:));
% 调用命令，并捕获输出
[status1, result1] = system(cmd1);
[status2, result2] = system(cmd2);
[status3, result3] = system(cmd3);
[status4, result4] = system(cmd4);
% 检查是否成功调用
if status1 == 0 && status2 == 0 && status3 == 0 && status4 == 0
    % 处理Python代码返回的结果
    trust(1) = str2double(result1);
    trust(2) = str2double(result2);
    trust(3) = str2double(result3);
    trust(4) = str2double(result4);
    % disp(result);
else
    error('调用Python代码失败');
end

%计算control performance
J_inf_max(time,:)=-lamda*(sum_cost'+trace(S_inf*Q))+trust;
randomJ(time)=J_inf_max(time,1);
[minJ(time),linkIndex(time)]=max(J_inf_max(time,:));
minAgeJ(time)=J_inf_max(time,tmpindex);
end

figure
t=1:1:MAXTIME;

plot(t,minJ,'r-o','LineWidth',1.6,'Color',[77 138 189]/255);% 
hold on
plot(t,minAgeJ,'g-x','LineWidth',1.6,'Color',[247 144 61]/255);% 
plot(t,randomJ,'b-','LineWidth',1.6,'Color',[89 169 90]/255);% 

fl = legend('Trust-AoI aware $J$','AoI-based $J$','Random $J$');
set(fl,'Interpreter','latex','Location','northeast','FontSize',12);
set(gca,'FontSize',14,'LineWidth',2)
xlabel('Time Sequences','Interpreter','LaTex'), ylabel('System Performance $J$','Interpreter','LaTex')
% ylim([0,500]), xlim([0,101])
%magnify(f2)
