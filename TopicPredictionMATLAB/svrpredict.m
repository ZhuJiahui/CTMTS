clear all;
clc;

%Y = [0.3, 0.4, 0.8, 0.7, 0.6];  % ¹Û²âÖµ
Y1 = load('TwData/norm_intensity.txt');
Y1 = Y1';
Y = Y1(1 : 10);
X = (1 : 10);

%entropy = [0.3, 0.4, 0.4, 0.4, 0.3];  % ìØÖµ£¨×´Ì¬2£©
entropy1 = load('TwData/norm_entropy.txt');
entropy1 = entropy1';
entropy = entropy1(1 : 10);

u = 0.7;
Trend1 = (Y1 -(1 - u) * entropy1) / u;
Trend = Trend1(1 : 10);


step = 5;

for i = 1 : step
    predicted_Y = stepwise_svr_p(Y', X');
    predicted_E = stepwise_svr_p(entropy', X');
    predicted_T = stepwise_svr_p(Trend', X');
    
    X = [X, (10 + i)];
    Y = [Y, predicted_Y];
    entropy = [entropy, predicted_E];
    Trend = [Trend, predicted_T];
end

residual_Y = Y1(11 : 15) - Y(11 : 15);
residual_E = entropy1(11 : 15) - entropy(11 : 15);
residual_T = Trend1(11 : 15) - Trend(11 : 15);

