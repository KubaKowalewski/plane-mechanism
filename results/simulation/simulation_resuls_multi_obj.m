%% Import Data

close all; clear all; clc;
sys_p = readtable("sim_results4");

%% Pareto search

% Find pareto optimal Range-Kmin by Class
pts = [1./sys_p.Mass,sys_p.Range,sys_p.k_min];
[ ~, idxs] = paretoFront(pts);
sys_p_pareto = sys_p(idxs,:);

%% Sorting

% Type Sorting
c1 = sys_p_pareto.Rw./sys_p_pareto.A > 0;
c2 = sys_p_pareto.Rw./sys_p_pareto.A < 0 & sys_p_pareto.Rh./sys_p_pareto.A > -1;
c3 = sys_p_pareto.Rw./sys_p_pareto.A < -1;
c1_color = [0 0.4470 0.7410];
c2_color = [0.8500 0.3250 0.0980];
c3_color = [0.9290 0.6940 0.1250];

% Symmetry Sorting
t1 = ismember(sys_p_pareto.Rr,1) & ismember(sys_p_pareto.Rf,"False");
t2 = sys_p_pareto.Rr < 1 & ismember(sys_p_pareto.Rf,"False");
t3 = sys_p_pareto.Rr < 1 & ismember(sys_p_pareto.Rf,"True");
t1_color = [0.4940 0.1840 0.5560];
t2_color = [0.4660 0.6740 0.1880];
t3_color = [0.3010 0.7450 0.9330];

%% Data Visualization

close all;

% Optimal Curves
v_color = [.7 .7 .7];

% Plotting parameters
ms = 100;
label_size = 20;
legend_size = 18;
title_size = 24;
axis_size = 16;
font_type = 'arial';
alpha = 0.05;
AZ = 50;
EL = 25;
fig  = figure(1); clf
fig.Position  = [100 100 2000 1200];

subplot(2,3,[1 2 4 5]); hold on; grid on
s = scatter3(sys_p.Mass,sys_p.Range,sys_p.k_min,ms,sys_p.k_min,'filled');
set(gca,'fontsize',axis_size)
colormap(turbo)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.Interpreter = 'latex';
cb.Label.String = '$\bar{K_{min}}$';
title("\bf{PL Performance Space}",'fontSize',title_size,'interpreter','latex')
xlabel("$\bar{M}$",'fontSize',label_size,'interpreter','latex')
ylabel("$\bar{R}$",'fontSize',label_size,'interpreter','latex')
zlabel("$\bar{K_{min}}$",'fontSize',label_size,'interpreter','latex')
set(gca,'fontname',font_type)
view([AZ EL])

subplot(2,3,3); hold on; grid on
scatter3(sys_p.Mass,sys_p.Range,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
scatter3(sys_p_pareto.Mass(c1),sys_p_pareto.Range(c1),sys_p_pareto.k_min(c1),ms,c1_color,'filled','DisplayName',"Type I")
scatter3(sys_p_pareto.Mass(c2),sys_p_pareto.Range(c2),sys_p_pareto.k_min(c2),ms,c2_color,'filled','DisplayName',"Type II")
scatter3(sys_p_pareto.Mass(c3),sys_p_pareto.Range(c3),sys_p_pareto.k_min(c3),ms,c3_color,'filled','DisplayName',"Type III")
set(gca,'fontsize',axis_size)
title("\bf{Pareto Optimal Set}",'fontSize',title_size,'interpreter','latex')
xlabel("$\bar{M}$",'fontSize',label_size,'interpreter','latex')
ylabel("$\bar{R}$",'fontSize',label_size,'interpreter','latex')
zlabel("$\bar{K_{min}}$",'fontSize',label_size,'interpreter','latex')
lg = legend('interpreter','latex');
lg.FontSize = legend_size;
set(lg,'Box','off')
view([AZ EL])

subplot(2,3,6); hold on; grid on
scatter3(sys_p.Mass,sys_p.Range,sys_p.k_min, ms,v_color,'filled','HandleVisibility','off','MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha)
scatter3(sys_p_pareto.Mass(t1),sys_p_pareto.Range(t1),sys_p_pareto.k_min(t1),ms,t1_color,'filled','DisplayName',"Symmetric")
scatter3(sys_p_pareto.Mass(t2),sys_p_pareto.Range(t2),sys_p_pareto.k_min(t2),ms,t2_color,'filled','DisplayName',"Asymmetric")
scatter3(sys_p_pareto.Mass(t3),sys_p_pareto.Range(t3),sys_p_pareto.k_min(t3),ms,t3_color,'filled','DisplayName',"Reflected")
set(gca,'fontsize',axis_size)
title("\bf{Pareto Optimal Set}",'fontSize',title_size,'interpreter','latex')
xlabel("$\bar{M}$",'fontSize',label_size,'interpreter','latex')
ylabel("$\bar{R}$",'fontSize',label_size,'interpreter','latex')
zlabel("$\bar{K_{min}}$",'fontSize',label_size,'interpreter','latex')
lg = legend('interpreter','latex');
lg.FontSize = legend_size;
set(lg,'Box','off')
view([AZ EL])

%% Objective function in Pareto Set

% Max Stiffness for each class
c1_idx = find(sys_p_pareto.k_min == max(sys_p_pareto.k_min(c1)));
c2_idx = find(sys_p_pareto.k_min == max(sys_p_pareto.k_min(c2)));
c3_idx = find(sys_p_pareto.k_min == max(sys_p_pareto.k_min(c3)));
combo = sys_p_pareto.k_min./max(sys_p_pareto.k_min) + 2*(sys_p_pareto.Range./max(sys_p_pareto.Range));
weighted = find(combo == max(combo));
disp(["C1", "Rw,Rh,Rr,Rf: ",sys_p_pareto.Rw(c1_idx),sys_p_pareto.Rh(c1_idx),sys_p_pareto.Rr(c1_idx),sys_p_pareto.Rf(c1_idx)])
disp(["C2", "Rw,Rh,Rr,Rf: ",sys_p_pareto.Rw(c2_idx),sys_p_pareto.Rh(c2_idx),sys_p_pareto.Rr(c2_idx),sys_p_pareto.Rf(c2_idx)])
disp(["C3", "Rw,Rh,Rr,Rf: ",sys_p_pareto.Rw(c3_idx),sys_p_pareto.Rh(c3_idx),sys_p_pareto.Rr(c3_idx),sys_p_pareto.Rf(c3_idx)])
disp(["Combo", "Rw,Rh,Rr,Rf: ",sys_p_pareto.Rw(weighted),sys_p_pareto.Rh(weighted),sys_p_pareto.Rr(weighted),sys_p_pareto.Rf(weighted)])


