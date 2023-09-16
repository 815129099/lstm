function SVMdianliyuce
tic;
close all;
clear;
clc;
format compact;

%%%%数据提取
N=0;
% data=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx');
data=xlsread('C:\Users\Administrator\Desktop\yuce\inputshuju.xlsx');
[m,n]=size(data);
%%功率数据与其他数据相比缺失4组，故除去最后四行
% m=m-4;
n=n-1;
data=data(:,1:n);
yuce_all=zeros((m-552),1);
yuce_begin=552;
yuce_end=m;
ts=data(:,n);
tsx=data(:,1:(n-1));


% 数据预处理,将原始数据进行归一化
ts = ts';
tsx = tsx';	
% 对ts进行归一化
[TS,TSps] = mapminmax(ts,1,2);
% 对TS进行转置,以符合libsvm工具箱的数据格式要求
TS = TS';
% 对tsx进行归一化
[TSX,TSXps] = mapminmax(tsx,1,2);	
TSX = TSX';


%%%数据的训练
  for yangben_number = yuce_begin:(yuce_end-1)
  N=N+1;
% 对归一化后的数据分训练样本和测试样本
 TS_xunlian = TS(1:yangben_number,:);
 TS_yuce = TS((yangben_number+1),:);
 TSX_xunlian = TSX(1:yangben_number,:);
 TSX_yuce = TSX((yangben_number+1),:);


%选择回归预测分析最佳的SVM参数c&g
% 粗略选择: 
 [bestmse,bestc,bestg] = SVMcgForRegress(TS_xunlian,TSX_xunlian,-8,8,-8,8);

%打印粗略选择结果
 disp('打印粗略选择结果');
 str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
 disp(str);

% 精细选择: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS_xunlian,TSX_xunlian,-5,5,-4,4,3,0.1,0.1,0.01);

% 打印精细选择结果
disp('打印精细选择结果');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

%% 分析最佳的参数c、g
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01 -t 2'];
model = svmtrain(TS_xunlian,TSX_xunlian,cmd);

%% SVM回归预测结果输出
IN=[TSX_xunlian;TSX_yuce];
OUT=[TS_xunlian;TS_yuce]; 
[predict,mse] = svmpredict(OUT,IN,model);
predict = mapminmax('reverse',predict',TSps);
predict = predict';
yuce_all(N,1)=predict(yangben_number+1,:);

% 打印回归结果
str = sprintf( '均方误差 MSE = %g 相关系数 R = %g%%',mse(2),mse(3)*100);
disp(str);

snapnow;
end


 
%%回归预测数据记录
YUCE=yuce_all
ts=ts';
ZHENGSHI=ts((yuce_begin+1):yuce_end,1)
%预测结果写入excel
xlswrite('YUCE.xlsx',YUCE);
xlswrite('ZHENGSHI.xlsx',ZHENGSHI);
xlswrite('ts.xlsx',ts);

%%预测结果作图分析

%%  1、预测结果图
figure;
plot(ts,'-go');
hold on;
x=(yuce_begin+1):1:m;
plot(x,yuce_all(:,1),'r-*');
hold off;
title('历史数据和回归预测数据对比','FontSize',12);
xlabel('天数(2015.5.28-2016.11.28)','FontSize',12);
ylabel('企业日平均用电功率大小','FontSize',12);
grid on;

%%  2、预测误差图
data2=ts(366:m,1);
yuce_error = yuce_all-data2;
train_error=zeros(yuce_begin,1);
figure;
plot([1:1:yuce_begin],train_error,'-r')
hold on
plot([(yuce_begin+1):1:yuce_end],yuce_error,'-ro')
title('预测结果误差图','FontSize',12);
xlabel('天数(2015.5.28-2016.11.28)','FontSize',12);
ylabel('误差量','FontSize',12);
grid on



%%%电流、电压和瞬时功率的监测
 % while(a>=0&&a<5)
  % close all;
 %  dataI=xlsread('dianliu.xlsx');
 %  dataU=xlsread('dianya.xlsx');
%   dataP=xlsread('gonglv.xlsx');
 %  plot(dataI(1,:),dataI(2,:),'-ro','LineWidth',1);
%   hold on
%   plot(dataU(1,:),dataU(2,:),'-g*','LineWidth',1);
%   hold on
 %  plot(dataP(1,:),dataP(2,:),'b-','LineWidth',1);
 %  legend('瞬时电流','瞬时电压','瞬时功率');
%   xlabel('时间','FontSize',12);
%   ylabel('电流/电压/瞬时功率值','FontSize',12);
%   title('瞬时电力参数监测');
%   grid on
 %  a=a+1;
 %  pause(5);
%  end







