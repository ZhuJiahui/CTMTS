clear all;
clc;

% using the lib-SVM
%Y = [0.3, 0.4, 0.8, 0.7, 0.6];  % �۲�ֵ
Y1 = load('TwData/norm_intensity.txt');
Y1 = Y1';
Y = Y1(1 : 10);

%entropy = [0.3, 0.4, 0.4, 0.4, 0.3];  % ��ֵ��״̬2��
entropy1 = load('TwData/norm_entropy.txt');
entropy1 = entropy1';
entropy = entropy1(1 : 10);

Anpn = [1.0, 0; 0, 1.0];  % ״̬ת�ƾ���
u = 0.5;  %0.8
Bn = [u, (1 - u)];  % ��������

Trend1 = (Y1 -(1 - u) * entropy1) / u;

delta = 0.01;  % 0.01
gamma = 0.1;
Qw = [delta^2 / 2, 0; 0, delta^2 / 4];  % ״̬����Э�������
Qv = delta ^ 2;  % �۲�ֵ����Э�������

% Xn|n-1^ ��ʼ��
Xnnm_e = zeros(2, 1);
Xnnm_e(1, 1) = (Y(1) - entropy(1) * (1 - u)) / u;
Xnnm_e(2, 1) = entropy(1);

step = 5;
future_step = 5;

% ���Ƶ�����ֵ��״̬1��
E_trend = zeros(1, (length(Y) + future_step));
E_entropy = zeros(1, (length(Y) + future_step));

all_Y = zeros(1, (length(Y) + future_step));
all_Y(1, 1 : length(Y)) = Y;
E_entropy(1, 1 : length(Y)) = entropy;

Pnnm = gamma ^ 2 * Qw;  % Ԥ�����Э�������

%% ����Ԥ��
for i = 1 : length(Y)
    KGn = Pnnm * Bn' * inv(Bn * Pnnm * Bn' + Qv);  % ��������Ϣ����
    ALPHAn = Y(i) - Bn * Xnnm_e;  % ��Ϣ���̾���
    Xnn_e = Xnnm_e + KGn * ALPHAn;
    
    E_trend(i) = Xnn_e(1, 1);
    
    %����
    if i < length(Y)
        slope1 = (Y(i + 1) - Y(i)) / 1;
        %slope2 = (entropy(i + 1) - entropy(i)) / 1;
        
        Xnnm_e(1, 1) = E_trend(i) + slope1;
        Xnnm_e(2, 1) = entropy(i + 1);
        
        Anpn(1, 1) = Xnnm_e(1, 1) / E_trend(i);
        Anpn(2, 2) = entropy(i + 1) / entropy(i);
    else
        t_svry = zeros(step, 1);
        t_svrx = zeros(step, 1);
        e_svry = zeros(step, 1);
        e_svrx = zeros(step, 1);
        for j = 1 : step
            t_svry(j) = E_trend(j + length(Y) - step);
            t_svrx(j) = j + length(Y) - step;
            e_svry(j) = entropy(j + length(Y) - step);
            e_svrx(j) = j + length(Y) - step;
        end
        Xnnm_e(1, 1) = stepwise_svr_p(t_svry, t_svrx);
        Xnnm_e(2, 1) = stepwise_svr_p(e_svry, e_svrx);
        
        Anpn(1, 1) = Xnnm_e(1, 1) / E_trend(i);
        Anpn(2, 2) = Xnnm_e(2, 1) / entropy(i);
    end
    
    Pnn = Pnnm - KGn * Bn * Pnnm;
    Pnnm = Anpn * Pnn * Anpn + Qw;
    
end

%% �ಽԤ��
for i = 1 : future_step
    all_Y(length(Y) + i) = Bn * Xnnm_e + normrnd(0, sqrt(Qv));
    
    KGn = Pnnm * Bn' * inv(Bn * Pnnm * Bn' + Qv);  % ��������Ϣ����
    ALPHAn = all_Y(length(Y) + i) - Bn * Xnnm_e;  % ��Ϣ���̾���
    Xnn_e = Xnnm_e + KGn * ALPHAn;
    
    E_trend(length(Y) + i) = Xnn_e(1, 1);
    E_entropy(length(Y) + i) = Xnn_e(2, 1);
    
    t_svry2 = E_trend(1 : (length(Y) + i))';
    t_svrx2 = (1 : (length(Y) + i))';
    e_svry2 = E_entropy(1 : (length(Y) + i))';
    e_svrx2 = (1 : (length(Y) + i))';

    Xnnm_e(1, 1) = stepwise_svr_p(t_svry2, t_svrx2);
    Xnnm_e(2, 1) = stepwise_svr_p(e_svry2, e_svrx2);
    
    Anpn(1, 1) = Xnnm_e(1, 1) / E_trend(length(Y) + i);
    Anpn(2, 2) = Xnnm_e(2, 1) / E_entropy(length(Y) + i);
    
    Pnn = Pnnm - KGn * Bn * Pnnm;
    Pnnm = Anpn * Pnn * Anpn + Qw;
    
end

residual_Y = Y1(11 : 15) - all_Y(11 : 15);
residual_E = entropy1(11 : 15) - E_entropy(11 : 15);
residual_T = Trend1(11 : 15) - E_trend(11 : 15);