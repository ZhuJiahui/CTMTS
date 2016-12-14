
%% a litte clean work
tic;
close all;
clear;
clc;
format compact;
%%

% 生成待回归的数据
x = (-1:0.1:1)';
y = -x.^2;

% 建模回归模型
model = svmtrain(y,x,'-s 3 -t 2 -c 5.0 -g 3.5 -p 0.001');  % e-SVR RBF 损失函数 核函数gamma函数设置 e-SVR损失函数p值

% 利用建立的模型看其在训练集合上的回归效果
[py,mse, dec_values] = svmpredict(y,x,model);
figure;
plot(x,y,'o');
hold on;
plot(x,py,'r*');
legend('原始数据','回归数据');
grid on;

% 进行预测
testx = 1.1;
%display('真实数据')
%testy = -testx.^2;

[ptesty,tmse, dec_values2] = svmpredict(0, testx, model);
display('预测数据');
ptesty
 
%%
toc