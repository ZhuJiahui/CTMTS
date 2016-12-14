clear all;
clc

Y1 = load('norm_intensity.txt');
Y1 = Y1';
entropy1 = load('norm_entropy.txt');
entropy1 = entropy1';
u = 0.8;
Trend1 = (Y1 -(1 - u) * entropy1) / u;
Trend1 = Trend1';