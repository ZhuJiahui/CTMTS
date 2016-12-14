clear all;
clc;
Y = [0.3, 0.4, 0.8, 0.7, 0.6];  % �۲�ֵ
entropy = [0.3, 0.4, 0.4, 0.4, 0.3];  % ��ֵ��״̬2��

Anpn = [1.0, 0; 0, 1.0];  % ״̬ת�ƾ���
u = 0.9;
Bn = [u, (1 - u)];  % ��������

delta = 0.05;
gamma = 0.1;
Qw = [delta^2 / 2, 0; 0, delta^2 / 4];  % ״̬����Э�������
Qv = delta ^ 2;  % �۲�ֵ����Э�������

% Xn|n-1^ ��ʼ��
Xnnm_e = zeros(2, 1); 
Xnnm_e(1, 1) = (Y(1) - entropy(1) * (1 - u)) / u;
Xnnm_e(2, 1) = entropy(1);

% ���Ƶ�����ֵ��״̬1��
E_trend = zeros(1, length(Y));
step = 5;

Pnnm = gamma ^ 2 * Qw;  % Ԥ�����Э�������

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
            t_svry(j) = E_trend(j + length(E_trend) - step);
            t_svrx(j) = j + length(E_trend) - step;
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
