%% Initialization
clc;
MAXTIME=100; % Maximum simulation time (number of timesteps)
LINKNUM=8; % Number of links
lamda=0.15; % Weighting factor
J_inf_max=99999*ones(MAXTIME,LINKNUM); % Maximum control performance
Eage=9999*ones(MAXTIME,LINKNUM); % Instantaneous age
trust=zeros(1,LINKNUM); % Trust values of links
linkIndex=zeros(MAXTIME,1); % Indices of links with maximum J_inf values
minJ=zeros(MAXTIME,1); % Minimum J_inf values over all the links
minAgeJ=zeros(MAXTIME,1); % Minimum J_inf values over links with minimum age
randomJ=zeros(MAXTIME,1); % Randomly selected J_inf values
aoi=zeros(LINKNUM,1); % Age of information for different links
input_data=readtable('./datasets/onlinedataset_sensor800.csv');
% Convert input table to matrix
input_data = table2array(input_data);
% Remove the first row and the first column from each dataset
input_data(1, :) = [];
input_data(:, 1) = [];
% Normalize the input data  
% input_data = normalize(input_data, 'range', [0, 1]); % Lack of precision 
input_data=(input_data - double(min(input_data(:)))) ./ (double(max(input_data(:))) - double(min(input_data(:))));
%% Control parameter determination
Wk = eye(2);
Uk = 1;
Q = [0.1 0;0 0.1];
R = 1;
A=[0.37 0;0.67 1];
B=[0.67 0.37]';
[K,S_inf,e]=dlqr(A,B,Wk,Uk);

%% Online processing
for time=1:MAXTIME
    % Simulate random AoI measurement
    Eage(time,:) = 10*rand(LINKNUM,1)*(1-eps);
    % Compute trust and age at current timestep
    sum_cost = zeros(LINKNUM,1);
    for j=1:LINKNUM 
        % Age of information for link j
        aoi(j) = floor(Eage(time,j));
        % Record the index of the link with minimum AoI
        [~, tmpindex] = min(aoi);
        for i=1:aoi(j)
            % Compute the cumulative cost
            sum_cost(j) = sum_cost(j) + trace(S_inf*A^i*Q*(A^i)');
        end
        % Call the pre-trained DNN model to predict the trust of each sensor channel pair
        % Construct the command to be executed, including the arguments to be passed
        cmd = sprintf('conda activate GANbp && python ./trust_predictor.py %f %f %f %f', input_data((j-1)*100+time,:));
        % Call the command and capture the output
        [status, result] = system(cmd);
        % Check if the call was successful
        if status == 0
            % Processing results
            trust(j) = str2double(result);
            % disp(result);
        else
            error('Failed to execute Python code.');
        end
    end
    % Compute control performance
    J_inf_max(time,:) = -lamda*(sum_cost' + trace(S_inf*Q)) + trust;
    randomJ(time) = J_inf_max(time,floor(8*rand*(1-eps))+1);
    [minJ(time), linkIndex(time)] = max(J_inf_max(time,:));
    minAgeJ(time) = J_inf_max(time,tmpindex);
end

figure(1)
title('trust-aoi-codesign')
t=1:1:MAXTIME;

plot(t,minJ,'r-o','LineWidth',1.6,'Color',[77 138 189]/255);
hold on
plot(t,minAgeJ,'g-x','LineWidth',1.6,'Color',[247 144 61]/255);
plot(t,randomJ,'b-','LineWidth',1.6,'Color',[89 169 90]/255);

fl = legend('Trust-AoI aware $J$','AoI-based $J$','Random $J$');
set(fl,'Interpreter','latex','Location','northeast','FontSize',12);
set(gca,'FontSize',14,'LineWidth',2)
xlabel('Time Sequences','Interpreter','LaTex'), ylabel('System Performance $J$','Interpreter','LaTex')
