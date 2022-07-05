%% Import Data

close all; clear all; clc;
sys_p = readtable("sim_results3");

%% Visualization

% Class Clustering
c1 = sys_p.Rh./sys_p.A > 0;
c2 = sys_p.Rh./sys_p.A < 0 & sys_p.Rh./sys_p.A > -1;
c3 = sys_p.Rh./sys_p.A < -1;
str = '#2c6e49';
c1_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#da344d';
c2_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#0582ca';
c3_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

% Type Clustering
t1 = ismember(sys_p.Rr,1) & ismember(sys_p.Rf,"False");
t2 = sys_p.Rr < 1 & ismember(sys_p.Rf,"False");
str = '#f8c630';
t1_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#724e91';
t2_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

% Plotting parameters
ms = 60;
label_size = 16;
legend_size = 14;
title_size = 18;

% 3D visualization of performance space
subplot(2,4,[1,2,5,6]); hold on; grid on;
s = scatter3(sys_p.Range,sys_p.Weight,sys_p.k_min,ms,sys_p.k_min,'filled');
title("PL SLM Performance Space",'Interpreter','latex','FontSize',title_size)
xlabel("$R$ (m)",'Interpreter','latex','FontSize',label_size); 
ylabel("$W$ (kg)",'Interpreter','latex','FontSize',label_size)
colormap(jet)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.String = '$K_{min}$ (N/m)';
cb.Label.Interpreter = 'latex';

subplot(2,4,3); hold on;
title("Range vs Stiffness",'Interpreter','latex','FontSize',title_size)
xlabel("$R$ (m)",'Interpreter','latex','FontSize',label_size); 
ylabel("$K_{min}$ (N/m)",'Interpreter','latex','FontSize',label_size)
scatter(sys_p.Range(c1),sys_p.k_min(c1),ms,c1_color,'filled','DisplayName','Class I');
scatter(sys_p.Range(c2),sys_p.k_min(c2),ms,c2_color,'filled','DisplayName','Class II');
scatter(sys_p.Range(c3),sys_p.k_min(c3),ms,c3_color,'filled','DisplayName','Class III');
lg = legend('Interpreter','latex');
lg.Interpreter = 'latex';
lg.FontSize = legend_size;
set(lg,'Box','off')

subplot(2,4,4); hold on;
title("Weight vs Stiffness",'Interpreter','latex','FontSize',title_size)
xlabel("$W$ (Kg)",'Interpreter','latex','FontSize',label_size); 
ylabel("$K_{min}$ (N/m)",'Interpreter','latex','FontSize',label_size)
scatter(sys_p.Weight(c1),sys_p.k_min(c1),ms,c1_color,'filled','DisplayName','Class I');
scatter(sys_p.Weight(c2),sys_p.k_min(c2),ms,c2_color,'filled','DisplayName','Class II');
scatter(sys_p.Weight(c3),sys_p.k_min(c3),ms,c3_color,'filled','DisplayName','Class III');
lg = legend('Interpreter','latex');
lg.Interpreter = 'latex';
lg.FontSize = legend_size;
set(lg,'Box','off')


subplot(2,4,7); hold on;
title("Range vs Stiffness",'Interpreter','latex','FontSize',title_size)
xlabel("$R$ (m)",'Interpreter','latex','FontSize',label_size); 
ylabel("$K_{min}$ (N/m)",'Interpreter','latex','FontSize',label_size)
scatter(sys_p.Range(t2),sys_p.k_min(t2),ms,t2_color,'filled','DisplayName','Type II');
scatter(sys_p.Range(t1),sys_p.k_min(t1),ms,t1_color,'filled','DisplayName','Type I');
lg = legend('Interpreter','latex');
lg.Interpreter = 'latex';
lg.FontSize = legend_size;
set(lg,'Box','off')

subplot(2,4,8); hold on;
title("Weight vs Stiffness",'Interpreter','latex','FontSize',title_size)
xlabel("$W$ (Kg)",'Interpreter','latex','FontSize',label_size); 
ylabel("$K_{min}$ (N/m)",'Interpreter','latex','FontSize',label_size)
scatter(sys_p.Weight(t2),sys_p.k_min(t2),ms,t2_color,'filled','DisplayName','Type II');
scatter(sys_p.Weight(t1),sys_p.k_min(t1),ms,t1_color,'filled','DisplayName','Type I');
lg = legend('Interpreter','latex');
lg.Interpreter = 'latex';
lg.FontSize = legend_size;
set(lg,'Box','off')

%% Visualization

figure(1); clf

% Class Clustering
c1 = sys_p.Rh./sys_p.A > 0;
c2 = sys_p.Rh./sys_p.A < 0 & sys_p.Rh./sys_p.A > -1;
c3 = sys_p.Rh./sys_p.A < -1;
str = '#2c6e49';
c1_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#da344d';
c2_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#0582ca';
c3_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

% Type Clustering
t1 = ismember(sys_p.Rr,1) & ismember(sys_p.Rf,"False");
t2 = sys_p.Rr < 1 & ismember(sys_p.Rf,"False");
str = '#f8c630';
t1_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;
str = '#724e91';
t2_color = sscanf(str(2:end),'%2x%2x%2x',[1 3])/255;

% Plotting parameters
ms = 60;
label_size = 16;
legend_size = 14;
title_size = 18;

% 3D visualization of performance space
subplot(1,3,1); hold on; grid on;
s = scatter3(sys_p.Weight,sys_p.Range,sys_p.k_min,ms,sys_p.k_min,'filled');
title("PL SLM Performance Space",'Interpreter','latex','FontSize',title_size)
xlabel("$R$ (m)",'Interpreter','latex','FontSize',label_size); 
ylabel("$W$ (kg)",'Interpreter','latex','FontSize',label_size)
colormap(jet)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.String = '$K_{min}$ (N/m)';
cb.Label.Interpreter = 'latex';

subplot(1,3,2); hold on;
s = scatter3(sys_p.Weight(c1),sys_p.Range(c1),sys_p.k_min(c1),ms,'filled');
s = scatter3(sys_p.Weight(c2),sys_p.Range(c2),sys_p.k_min(c2),ms,'filled');
s = scatter3(sys_p.Weight(c3),sys_p.Range(c3),sys_p.k_min(c3),ms,'filled');

subplot(1,3,3); hold on;
s = scatter3(sys_p.Weight(t1),sys_p.Range(t1),sys_p.k_min(t1),ms,'filled');
s = scatter3(sys_p.Weight(t2),sys_p.Range(t2),sys_p.k_min(t2),ms,'filled');

%%
k = boundary(sys_p.Weight,sys_p.Range,sys_p.k_min);

%%
figure(2); clf
s = scatter3(sys_p.Weight(k),sys_p.Range(k),sys_p.k_min(k));



