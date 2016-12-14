function predicted_y = stepwise_svr_p(y, x)

% 建模回归模型
model = svmtrain(y,x,'-s 3 -t 2 -c 0.5 -g 2.5 -p 0.005');  % e-SVR RBF 损失函数 核函数gamma函数设置 e-SVR损失函数p值
%model = svmtrain(y, x, 'showplot', true, 'kernel_function', 'rbf');  % e-SVR RBF 损失函数 核函数gamma函数设置 e-SVR损失函数p值


% 利用建立的模型看其在训练集合上的回归效果
%[predicted_label, accuracy, prob_estimates] = svmpredict(y, x, model);
% accuracy: accuracy, mean squared error, squared correlation coefficient

%figure;
%plot(x, y, 'o');
%hold on;
%plot(x, predicted_label, 'r*');
%legend('原始数据','回归数据');
%grid on;

% 进行预测
testx = x(end) + 1;
%display('真实数据')
%testy = -testx.^2;

[predicted_label2, accuracy2, prob_estimates2] = svmpredict(0, testx, model);
%display('预测数据');
predicted_y = predicted_label2;

%%

end