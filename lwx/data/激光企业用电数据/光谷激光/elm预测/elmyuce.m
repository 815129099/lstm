tic;
close all;
clear;
clc;
format compact;

%%%%������ȡ
N=0;
data=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\input.xlsx');
[m,n]=size(data);
%%���������������������ȱʧ4�飬�ʳ�ȥ�������
m=m-4;
data=data(1:m,:);
yuce_all=zeros((m-366),1);
yuce_begin=365;
yuce_end=m;
T=data(:,n);
T=T';
P=data(:,:);
P=P';

% [T,Tps]=mapminmax(t,0,1);   
% T=T';
%[P,Pps]=mapminmax(p,0,1);
%P=P';


 for  yangben_number = yuce_begin:(yuce_end-1)
 N=N+1;
  % �Թ�һ��������ݷ�ѵ�������Ͳ�������
  P_train=P(:,1:yangben_number);
  P_test=P(:,(yangben_number+1));
  T_train=T(:,1:yangben_number);
  T_test=T(:,(yangben_number+1));
  
  [Pn_train,inputps]=mapminmax(P_train,0,1);
  Pn_test=mapminmax('apply',P_test,inputps);
  [Tn_train,outputps]=mapminmax(T_train,0,1);
  Tn_test=mapminmax('apply',T_test,outputps);
  
  %%ELM�ع�Ԥ��
  [IW,B,LW,TF,TYPE]=elmtrain(Pn_train,Tn_train,100,'sig',0);
  Tn_yuce=elmpredict(Pn_test,IW,B,LW,TF,TYPE);
  T_yuce=mapminmax('reverse',Tn_yuce,outputps);
  T_yuce=T_yuce';
 yuce_all(N,1)=T_yuce;
  
 
  %%Ԥ�����Ա�
  % result=[T((yuce_begin+1):(yuce_begin+N),1) yuce_all((yuce_begin+1):(yuce_begin+N),1)];
 % E=mse(yuce_all-T_test);
  M=length(yuce_all);
 % R2=(M*sum(T_yuce.*T_test)-sum(T_yuce)*sum(T_test))^2/((M*sum((T_yuce).^2)-(sum(T_yuce))^2)*(M*sum((T_test).^2)-(sum(T_test))^2));
  % ��ӡ�ع���
% str = sprintf( '������� MSE = %g ���ϵ�� R = %g%%',E,R2);
% disp(str);

   snapnow
 end 
  
  %%��������
  data_zhengshi=T(:,(yuce_begin+1):m)';
  E=mse(yuce_all-data_zhengshi);
  str = sprintf( '������� MSE = %g ',E)

 

  
  %%Ԥ��Ա���ͼ
    data1(:,1)=yuce_all;
  data1(:,2)=data(366:m,n);
  data1;
  plot(data1(:,1),'-bo')
  hold on
  plot(data1(:,2),'-r*')
  grid on
  
  

