close all
clear
clc

% Ectract data from txt file
data = readmatrix('Numbers.txt');
x = data(1);
y = data(2);
z = data(3);

% Plot data
figure(1)
hold on; grid on; box on;
plot3(x, y, z, 'ro', 'MarkerSize', 8, 'LineWidth', 2);
title('Single 3D Point from TXT File');