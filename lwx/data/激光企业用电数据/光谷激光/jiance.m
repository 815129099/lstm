clc;
clear all;

%%ƽ���й�����/ʱ��ƽ���޹�����/ʱ������������ļ��
%%�й����޹��ļ��
A=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\����.xlsx');
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
title('�人��ȼ��⼼���ɷ����޹�˾�õ�ƽ���й�����','FontSize',12);
xlabel('��һ����/ʱ','FontSize',12);
ylabel('���ʴ�С','FontSize',12);
grid on

figure
plot(B((m-720):m,2),'-go');
title('�人��ȼ��⼼���ɷ����޹�˾�õ�ƽ���޹�����','FontSize',12);
xlabel('��һ����/ʱ','FontSize',12);
ylabel('���ʴ�С','FontSize',12);
grid on

%%����������ļ��
C=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\����.xlsx');
C=C(:,4);
xlswrite('C:\Users\Administrator\Desktop\������ѹ��ֵ\shuju.xlsx',C,'A2:A553');
figure;
plot(C,'-bo');
title('�人��ȼ��⼼���ɷ����޹�˾�õ��������','FontSize',12);
xlabel('����(2015.5.28-2016.11.28)','FontSize',12);
ylabel('������С','FontSize',12);
grid on


 %%���������ѹ�ļ��
clc;
clear all;
A=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\����.xlsx');
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
legend('A�����','C�����');
grid on
title('�人��ȼ��⼼���ɷ����޹�˾�����ܱ������','FontSize',12);
xlabel('��һ����/ʱ','FontSize',12);
ylabel('������С','FontSize',12);


clc;
clear all;
A=xlsread('C:\Users\Administrator\Desktop\LV���\������ҵ�õ�����\��ȼ���\��ѹ.xlsx');
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
xlswrite('C:\Users\Administrator\Desktop\������ѹ��ֵ\shuju.xlsx',B,'D2:E553');
figure;
plot(B((m-720):m,1),'r-');
hold on
plot(B((m-720):m,2),'b-');
legend('A���ѹ','C���ѹ');
grid on
title('�人��ȼ��⼼���ɷ����޹�˾�����ܱ����ѹ','FontSize',12);
xlabel('��һ����/ʱ','FontSize',12);
ylabel('��ѹ��С','FontSize',12);

