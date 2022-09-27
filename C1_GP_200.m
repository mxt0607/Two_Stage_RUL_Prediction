%Cell 1
%load relevant files
clc
clear all;
load('Capacity data.mat')
a2 = a2';
EOM = 200;
M = length(a2)-EOM;
%Training set of the GPR model 
X_train = 1:EOM;                                 %X_train is the input of the model
X_train = X_train';
Y_train = a2(1:EOM);                             %Y_train is the corresponding capacity of the cell

%Testing set of the GPR model
future_cycle = 5;                                  % For future inference
X_test = EOM+1:length(a2)+future_cycle;            % Testing input 
X_test = X_test';                                  
Y_test = a2(EOM+1:length(a2));                     % Testing capacity

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Proposed method
lik = @likGauss; 
inf = @infExact;
cov = @covSEiso; hyp1.cov = [3.5 -5.3650];
sn = 0.1; hyp1.lik =log(sn); 
mean_func1 = {@meanSum, {@meanExp, @meanConst}}; hyp1.mean = [[-0.0002079, 0.003009, 1.085, -3.117e-05]'; 0];  % The parameters are identified by the cftool
model1 = minimize(hyp1, @gp, -1000, inf, mean_func1, cov, lik, X_train, Y_train);    

%Capacity estimation of the testing cycles 
[GP_test_est_mean1, GP_test_est_var1] = gp(model1,inf, mean_func1, cov, lik, X_train, Y_train, X_test);  
[Y_train_est_mean1, Y_train_est_var1] = gp(model1,inf, mean_func1, cov, lik, X_train, Y_train, X_train);  
GP_mean1 = [Y_train_est_mean1; GP_test_est_mean1]; 
GP_interval_up1 = [Y_train_est_mean1; GP_test_est_mean1+2.576*sqrt(GP_test_est_var1)];
GP_interval_low1 = [Y_train_est_mean1; GP_test_est_mean1-2.576*sqrt(GP_test_est_var1)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Without mean function
lik2 = @likGauss; 
inf2 = @infExact;
cov2 = @covSEiso; hyp2.cov = [3.5 -5.3650];
sn = 0.1; hyp2.lik =log(sn); 
mean_func2 = @meanZero; hyp.mean = [];
model2 = minimize(hyp2, @gp, -1000, inf2, mean_func2, cov2, lik2, X_train, Y_train);    

%Capacity Estimation of the testing cycles 
[GP_test_est_mean2,GP_test_est_var2] = gp(model2, inf2, mean_func2, cov2, lik2, X_train, Y_train, X_test);  
[Y_train_est_mean2,Y_train_est_var2] = gp(model2, inf2, mean_func2, cov2, lik2, X_train, Y_train, X_train);  
GP_mean2 = [Y_train_est_mean2; GP_test_est_mean2]; 
GP_interval_up2 = [Y_train_est_mean2; GP_test_est_mean2+2.576*sqrt(GP_test_est_var2)];
GP_interval_low2 = [Y_train_est_mean2; GP_test_est_mean2-2.576*sqrt(GP_test_est_var2)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Without bias
lik3 = @likGauss; 
inf3 = @infExact;
cov3 = @covSEiso; hyp3.cov = [3.5 -5.3650];
sn = 0.1; hyp3.lik =log(sn); 
mean_func3 = @meanExp; hyp3.mean = [-0.0002079, 0.003009, 1.085, -3.117e-05]';
model3 = minimize(hyp3, @gp, -1000, inf3, mean_func3, cov3, lik3, X_train, Y_train);    

%Capacity Estimation of the testing cycles 
[GP_test_est_mean3, GP_test_est_var3] = gp(model3, inf3, mean_func3, cov3, lik3, X_train, Y_train, X_test);  
[Y_train_est_mean3, Y_train_est_var3] = gp(model3, inf3, mean_func3, cov3, lik3, X_train, Y_train, X_train);  
GP_mean3 = [Y_train_est_mean3; GP_test_est_mean3]; 
GP_interval_up3 = [Y_train_est_mean3; GP_test_est_mean3+2.576*sqrt(GP_test_est_var3)];
GP_interval_low3 = [Y_train_est_mean3; GP_test_est_mean3-2.576*sqrt(GP_test_est_var3)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure()
f = [GP_interval_up1; flipdim(GP_interval_low1,1)];  
h = fill([[X_train; X_test]; flipdim([X_train; X_test],1)], f, [1 0.8 0.8]);
set(h,'edgecolor','white');
hold on;
plot(GP_mean1, 'Color', [0,0,1], 'LineWidth',2)
hold on;
plot(GP_mean2, 'Color', [0.69,0.09,0.12], 'LineWidth',2)
hold on;
plot(GP_mean3, 'Color', [1,0.5,0], 'LineWidth',2)
hold on;
plot(a2, 'k','LineWidth',2)
hold on;
plot([500, 2300], [0.88, 0.88], '-.', 'Color', [1,0.27,0], 'LineWidth', 1)
hold on;
plot([EOM, EOM], [0.8, 1.099], '--', 'Color', [0.5,0.5,0.5], 'LineWidth', 1)
xlabel('\fontsize{20}Cycle Number');
ylabel('\fontsize{20}Capacity (Ah)');
ylim([0.8,1.1])
xlim([0,2500])
text(EOM+2, 0.85, ['EOM=',num2str(EOM)],'Color','K', 'FontSize', 14)
text(500, 0.89, 'EOL=0.88 Ah', 'Color', 'K','FontSize', 14)
set(gca, 'FontSize', 15)
title (['\fontsize{25}Cell 1: EOM=', num2str(EOM)])
lgd = legend({'\fontsize{12} 99% confidence interval','\fontsize{12} GPR w/ prior and bias', '\fontsize{12} GPR w/o prior','\fontsize{12} GPR w/o bias', '\fontsize{13} Measured capacity'}, 'Box', 'off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Fitting curve%%% 
g=@(x,k)(x(1)*exp(x(2)*k)+x(3)*exp(x(4)*k));
paras = [-0.0002079, 0.003009, 1.085, -3.117e-05];
for k=1:length(a1)
    iden(k) = g(paras, k);
end  
iden = iden';
figure()
hold on;
scatter(1:length(a1),a1,30,'k')
hold on;
plot(iden, 'Color','r','LineWidth',2)
grid on ; 
xlabel('\fontsize{30}Cycle Number');
ylabel('\fontsize{30}Capacity (Ah)');
ylim([0.88,1.1])
xlim([0,2500])
set(gca,'FontSize',30)
title ('\fontsize{35}a-1')
lgd = legend({'\fontsize{25} Actual capacity','\fontsize{25} Fitting curve'},'Box','off');
set(gca,'LineWidth',1.5);
box on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Prior LIB and test LIB%%% 

figure()
hold on;
scatter(1:length(a2),a2,30,'k')
hold on;
scatter(1:length(a1),a1,30,'b')
grid on ; 
xlabel('\fontsize{30}Cycle Number');
ylabel('\fontsize{30}Capacity (Ah)');
ylim([0.88,1.1])
xlim([0,2500])
set(gca,'FontSize',25)
%title ('\fontsize{35} Cell 1')
lgd = legend({'\fontsize{25} Cell 1','\fontsize{25} Prior LIB (a-1)'},'Box','off');
set(gca,'LineWidth',1.5);
box on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%proposed%%% 
Pre_RUL_mean1 = find(GP_mean1<=0.88,1)-EOM;
Pre_RUL_up1 = find(GP_interval_up1<=0.88,1)-EOM;
Pre_RUL_low1 = find(GP_interval_low1<=0.88,1)-EOM;
RUL_interval1 = Pre_RUL_up1-Pre_RUL_mean1;
Actual_RUL1 = length(a2)-EOM;
EM1 = abs(Pre_RUL_mean1-Actual_RUL1);
AM1 = (1 - EM1/Actual_RUL1)*100;
AM_interval1 = (RUL_interval1/Actual_RUL1)*100;
SM1 = sqrt(mean((a2(EOM+1:length(a2))-GP_mean1(EOM+1:length(a2))).^2));

%%% w/o mean function%%% 
Pre_RUL_mean2 = find(GP_mean2<=0.88,1)-EOM;
Pre_RUL_up2 = find(GP_interval_up2<=0.88,1)-EOM;
Pre_RUL_low2 = find(GP_interval_low2<=0.88,1)-EOM;
RUL_interval2 = Pre_RUL_up2-Pre_RUL_mean2;
Actual_RUL2 = length(a2)-EOM;
EM2 = abs(Pre_RUL_mean2-Actual_RUL2);
AM2 = (1 - EM2/Actual_RUL2)*100;
AM_interval2 = (RUL_interval2/Actual_RUL2)*100;
SM2 = sqrt(mean((a2(EOM+1:length(a2))-GP_mean2(EOM+1:length(a2))).^2));

%%% w/o bias%%% 
Pre_RUL_mean3 = find(GP_mean3<=0.88,1)-EOM;
Pre_RUL_up3 = find(GP_interval_up3<=0.88,1)-EOM;
Pre_RUL_low3 = find(GP_interval_low3<=0.88,1)-EOM;
RUL_interval3 = Pre_RUL_up3-Pre_RUL_mean3;
Actual_RUL3 = length(a2)-EOM;
EM3 = abs(Pre_RUL_mean3-Actual_RUL3);
AM3 = (1 - EM3/Actual_RUL3)*100;
AM_interval3 = (RUL_interval3/Actual_RUL3)*100;
SM3 = sqrt(mean((a2(EOM+1:length(a2))-GP_mean3(EOM+1:length(a2))).^2));

%%% CNN direct%%% 
Pre_RUL_mean4 = 2092-EOM;
Actual_RUL4 = length(a2)-EOM;
EM4 = abs(Pre_RUL_mean4-Actual_RUL4);
AM4 = (1 - EM4/Actual_RUL4)*100;




