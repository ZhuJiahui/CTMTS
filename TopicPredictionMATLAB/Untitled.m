
%% a litte clean work
tic;
close all;
clear;
clc;
format compact;
%%

% ���ɴ��ع������
x = (-1:0.1:1)';
y = -x.^2;

% ��ģ�ع�ģ��
model = svmtrain(y,x,'-s 3 -t 2 -c 5.0 -g 3.5 -p 0.001');  % e-SVR RBF ��ʧ���� �˺���gamma�������� e-SVR��ʧ����pֵ

% ���ý�����ģ�Ϳ�����ѵ�������ϵĻع�Ч��
[py,mse, dec_values] = svmpredict(y,x,model);
figure;
plot(x,y,'o');
hold on;
plot(x,py,'r*');
legend('ԭʼ����','�ع�����');
grid on;

% ����Ԥ��
testx = 1.1;
%display('��ʵ����')
%testy = -testx.^2;

[ptesty,tmse, dec_values2] = svmpredict(0, testx, model);
display('Ԥ������');
ptesty
 
%%
toc