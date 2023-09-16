clc;
clear all;
data1=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\冻结表码');
data2=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\需量');
data3=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\功率');
data4=xlsread('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\温度');
data3=data3(1:896,:);
%%%将电力参数汇总到一个表格input.xls
%%总有功电能
[m,n]=size(data1);
data1=data1(:,[4,6,7,8,9]);
xlswrite('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx',data1,'A1:E552');

%%最大需量
data2=data2(:,4);
xlswrite('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx',data2,'F1:F552');

%%温度
xlswrite('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx',data4,'G1:H552');

%%有功功率和无功功率
A=data3(:,1:24);
[m,n]=size(A);
 A(isnan(A))=0;
 B=[];
for j=1:m;
s=0;
for i=1:n;
s=s+A((i-1)*m+j);
end
B(j,1)=s/n;
end

data3_pingjun=B;

data3_yougong=[];
data3_wugong=[];
c=1;
d=1;
for i=1:m
    if(rem(i,2)==1)
       data3_yougong(c,1)=roundn(data3_pingjun(i,1),-2);
       c=c+1;
    else
        data3_wugong(d,1)=roundn(data3_pingjun(i,1),-2); 
        d=d+1;
    end
end
  %%将平均有功功率和平均无功功率存入input表格
     xlswrite('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx',data3_wugong,'I1:I448');
     xlswrite('C:\Users\Administrator\Desktop\LV设计\激光企业用电数据\光谷激光\input.xlsx',data3_yougong,'J1:J448');
     





