clc;
clear all;

%%平均有功功率/时、平均无功功率/时和日最大需量的监测
%%有功和无功的监测
A=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\功率.xlsx');
A(isnan(A))=0;
[m,n]=size(A);
n=n-2;
A=A(1:m,1:n);
B=[];
for i=1:fix(m/2)
    for j=1:n
        B(n*i+j,1)=A(2*(i-1)+1,j);
        B(n*i+j,2)=A(2*i,j);
    end
end
figure;
plot(B((m-720):m,1),'-g*');
title('武汉光谷激光技术股份有限公司用电平均有功功率','FontSize',12);
xlabel('近一个月/时','FontSize',12);
ylabel('功率大小','FontSize',12);
grid on

figure
plot(B((m-720):m,2),'-go');
title('武汉光谷激光技术股份有限公司用电平均无功功率','FontSize',12);
xlabel('近一个月/时','FontSize',12);
ylabel('功率大小','FontSize',12);
grid on

%%日最大需量的监测
C=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\需量.xlsx');
C=C(:,4);
xlswrite('C:\Users\Administrator\Desktop\电流电压插值\shuju.xlsx',C,'A2:A553');
figure;
plot(C,'-bo');
title('武汉光谷激光技术股份有限公司用电最大需量','FontSize',12);
xlabel('天数(2015.5.28-2016.11.28)','FontSize',12);
ylabel('需量大小','FontSize',12);
grid on


 %%两相电流电压的监测
clc;
clear all;
A=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\电流.xlsx');
A(isnan(A))=0;
[m,n]=size(A);
n=n-2;
A=A(1:m,1:n);
B=[];
for i=0:fix((m-1)/3)
    for j=1:n
        B(n*i+j,1)=A(3*i+1,j);
         B(n*i+j,1)=roundn( B(n*i+j,1),-2);
    end
end
for i=0:fix((m-3)/3)
    for j=1:n
        B(n*i+j,2)=A(3*i+3,j);
         B(n*i+j,2)=roundn( B(n*i+j,2),-2);
    end
end
[m,n]=size(B);
figure;
plot(B((m-720):m,1),'r-');
hold on
plot(B((m-720):m,2),'b-');
legend('A相电流','C相电流');
grid on
title('武汉光谷激光技术股份有限公司供电总表相电流','FontSize',12);
xlabel('近一个月/时','FontSize',12);
ylabel('电流大小','FontSize',12);


clc;
clear all;
A=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\电压.xlsx');
A(isnan(A))=0;
[m,n]=size(A);
n=n-2;
A=A(1:m,1:n);
B=[];
for i=0:fix((m-1)/3)
    for j=1:n
        B(n*i+j,1)=A(3*i+1,j);
         B(n*i+j,1)=roundn( B(n*i+j,1),-2);
    end
end
for i=0:fix((m-3)/3)
    for j=1:n
        B(n*i+j,2)=A(3*i+3,j);
         B(n*i+j,2)=roundn( B(n*i+j,2),-2);
    end
end
[m,n]=size(B);
xlswrite('C:\Users\Administrator\Desktop\电流电压插值\shuju.xlsx',B,'D2:E553');
figure;
plot(B((m-720):m,1),'r-');
hold on
plot(B((m-720):m,2),'b-');
legend('A相电压','C相电压');
grid on
title('武汉光谷激光技术股份有限公司供电总表相电压','FontSize',12);
xlabel('近一个月/时','FontSize',12);
ylabel('电压大小','FontSize',12);

