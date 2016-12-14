clear all;
clc;

intensity = load('TwData/selected_intensity.txt');
entropy = load('TwData/selected_entropy.txt');

intensity2 = log10(intensity + 1);
max_i = max(intensity2);
min_i = min(intensity2);
norm_intensity = (intensity2 - min_i + 0.1) / (max_i - min_i + 0.1);

entropy = entropy .^ (-1);
max_e = max(entropy);
min_e = min(entropy);
norm_entropy = (entropy - min_e + 0.1) / (max_e - min_e + 0.1);

dlmwrite('TwData/norm_intensity.txt', norm_intensity);
dlmwrite('TwData/norm_entropy.txt', norm_entropy);