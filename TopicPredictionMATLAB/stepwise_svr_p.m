function predicted_y = stepwise_svr_p(y, x)

% ��ģ�ع�ģ��
model = svmtrain(y,x,'-s 3 -t 2 -c 0.5 -g 2.5 -p 0.005');  % e-SVR RBF ��ʧ���� �˺���gamma�������� e-SVR��ʧ����pֵ
%model = svmtrain(y, x, 'showplot', true, 'kernel_function', 'rbf');  % e-SVR RBF ��ʧ���� �˺���gamma�������� e-SVR��ʧ����pֵ


% ���ý�����ģ�Ϳ�����ѵ�������ϵĻع�Ч��
%[predicted_label, accuracy, prob_estimates] = svmpredict(y, x, model);
% accuracy: accuracy, mean squared error, squared correlation coefficient

%figure;
%plot(x, y, 'o');
%hold on;
%plot(x, predicted_label, 'r*');
%legend('ԭʼ����','�ع�����');
%grid on;

% ����Ԥ��
testx = x(end) + 1;
%display('��ʵ����')
%testy = -testx.^2;

[predicted_label2, accuracy2, prob_estimates2] = svmpredict(0, testx, model);
%display('Ԥ������');
predicted_y = predicted_label2;

%%

end