%% min stiffness

close all; clear all; clc;
sys_p = readtable("pareto_front_all");

% apply scaling factors
SD = 1;
SL = 1;

% scale for length
sys_p.L = sys_p.L .* SL;
sys_p.k_max = sys_p.k_max .* 1/SL;
sys_p.k_min = sys_p.k_min .* 1/SL;
sys_p.W = sys_p.W .* SL;
sys_p.A = sys_p.A.*SL;
sys_p.B = sys_p.B.*SL;
sys_p.C = sys_p.C.*SL;
% scale for diameter
sys_p.k_max = sys_p.k_max .* SD^2;
sys_p.k_min = sys_p.k_min .* SD^2;
sys_p.W = sys_p.W .* SD^2;

% Set mechanism spaces to filter
filter = ["1x","1y","1z","2x","2y","2z","3x","3y","3z"];
sys_p=sys_p(ismember(sys_p.ID,filter),:);

% Looking at min stiffness of mechanism
figure(1); clf; hold on
s = scatter3(sys_p.L,sys_p.W,sys_p.k_min,[],sys_p.k_min,'filled','SizeData',50);
xlabel("Range (m)")
ylabel("Weight (kg)")
zlabel(" Min Stiffness (N/m)")
title("SLM System Property Sweep (K min)")
set(gca,'fontsize',20)
grid on

% Find best SLMs
idx = find(sys_p.k_max==max(sys_p.k_max));
fprintf('(Best Max Stiffness) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))

idx = find(sys_p.k_min==max(sys_p.k_min));
fprintf('(Best Min Stiffness) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))

idx = find((sys_p.L./sys_p.W)==max(sys_p.L./sys_p.W));
fprintf('(Best Range/Weight Ratio) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))
