function SVMdianliyuce
tic;
close all;
clear;
clc;
format compact;

%%%%������ȡ
N=0;
% data=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\input.xlsx');
data=xlsread('C:\Users\Administrator\Desktop\yuce\inputshuju.xlsx');
[m,n]=size(data);
%%���������������������ȱʧ4�飬�ʳ�ȥ�������
% m=m-4;
n=n-1;
data=data(:,1:n);
yuce_all=zeros((m-552),1);
yuce_begin=552;
yuce_end=m;
ts=data(:,n);
tsx=data(:,1:(n-1));


% ����Ԥ����,��ԭʼ���ݽ��й�һ��
ts = ts';
tsx = tsx';	
% ��ts���й�һ��
[TS,TSps] = mapminmax(ts,1,2);
% ��TS����ת��,�Է���libsvm����������ݸ�ʽҪ��
TS = TS';
% ��tsx���й�һ��
[TSX,TSXps] = mapminmax(tsx,1,2);	
TSX = TSX';


%%%���ݵ�ѵ��
  for yangben_number = yuce_begin:(yuce_end-1)
  N=N+1;
% �Թ�һ��������ݷ�ѵ�������Ͳ�������
 TS_xunlian = TS(1:yangben_number,:);
 TS_yuce = TS((yangben_number+1),:);
 TSX_xunlian = TSX(1:yangben_number,:);
 TSX_yuce = TSX((yangben_number+1),:);


%ѡ��ع�Ԥ�������ѵ�SVM����c&g
% ����ѡ��: 
 [bestmse,bestc,bestg] = SVMcgForRegress(TS_xunlian,TSX_xunlian,-8,8,-8,8);

%��ӡ����ѡ����
 disp('��ӡ����ѡ����');
 str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
 disp(str);

% ��ϸѡ��: 
[bestmse,bestc,bestg] = SVMcgForRegress(TS_xunlian,TSX_xunlian,-5,5,-4,4,3,0.1,0.1,0.01);

% ��ӡ��ϸѡ����
disp('��ӡ��ϸѡ����');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

%% ������ѵĲ���c��g
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01 -t 2'];
model = svmtrain(TS_xunlian,TSX_xunlian,cmd);

%% SVM�ع�Ԥ�������
IN=[TSX_xunlian;TSX_yuce];
OUT=[TS_xunlian;TS_yuce]; 
[predict,mse] = svmpredict(OUT,IN,model);
predict = mapminmax('reverse',predict',TSps);
predict = predict';
yuce_all(N,1)=predict(yangben_number+1,:);

% ��ӡ�ع���
str = sprintf( '������� MSE = %g ���ϵ�� R = %g%%',mse(2),mse(3)*100);
disp(str);

snapnow;
end


 
%%�ع�Ԥ�����ݼ�¼
YUCE=yuce_all
ts=ts';
ZHENGSHI=ts((yuce_begin+1):yuce_end,1)
%Ԥ����д��excel
xlswrite('YUCE.xlsx',YUCE);
xlswrite('ZHENGSHI.xlsx',ZHENGSHI);
xlswrite('ts.xlsx',ts);

%%Ԥ������ͼ����

%%  1��Ԥ����ͼ
figure;
plot(ts,'-go');
hold on;
x=(yuce_begin+1):1:m;
plot(x,yuce_all(:,1),'r-*');
hold off;
title('��ʷ���ݺͻع�Ԥ�����ݶԱ�','FontSize',12);
xlabel('����(2015.5.28-2016.11.28)','FontSize',12);
ylabel('��ҵ��ƽ���õ繦�ʴ�С','FontSize',12);
grid on;

%%  2��Ԥ�����ͼ
data2=ts(366:m,1);
yuce_error = yuce_all-data2;
train_error=zeros(yuce_begin,1);
figure;
plot([1:1:yuce_begin],train_error,'-r')
hold on
plot([(yuce_begin+1):1:yuce_end],yuce_error,'-ro')
title('Ԥ�������ͼ','FontSize',12);
xlabel('����(2015.5.28-2016.11.28)','FontSize',12);
ylabel('�����','FontSize',12);
grid on



%%%��������ѹ��˲ʱ���ʵļ��
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
 %  legend('˲ʱ����','˲ʱ��ѹ','˲ʱ����');
%   xlabel('ʱ��','FontSize',12);
%   ylabel('����/��ѹ/˲ʱ����ֵ','FontSize',12);
%   title('˲ʱ�����������');
%   grid on
 %  a=a+1;
 %  pause(5);
%  end







