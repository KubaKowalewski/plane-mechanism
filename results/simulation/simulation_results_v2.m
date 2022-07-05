%% Import Data

close all; clear all; clc;
sys_p = readtable("sim_results3");

%% Pareto search
pts = [sys_p.Range,sys_p.k_min];
[ ~, idxs_rk] = paretoFront(pts);

pts = [sys_p.Weight,sys_p.k_min];
[ ~, idxs_wk] = paretoFront(pts);


%%
close all;

% Class Clustering
c1 = sys_p.Rh./sys_p.A > 0;
c2 = sys_p.Rh./sys_p.A < 0 & sys_p.Rh./sys_p.A > -1;
c3 = sys_p.Rh./sys_p.A < -1;
c1_color = [0 0.4470 0.7410];
c2_color = [0.8500 0.3250 0.0980];
c3_color = [0.9290 0.6940 0.1250];

% Type Clustering
t1 = ismember(sys_p.Rr,1) & ismember(sys_p.Rf,"False");
t2 = sys_p.Rr < 1 & ismember(sys_p.Rf,"False");
t1_color = [0.4940 0.1840 0.5560];
t2_color = [0.4660 0.6740 0.1880];

% Optimal Curves
v_color = [.7 .7 .7];

% Plotting parameters
ms = 30;
label_size = 20;
legend_size = 18;
title_size = 24;
axis_size = 16;
font_type = 'arial';

figure(1); clf

subplot(2,3,1); hold on
s = scatter3(sys_p.Weight,sys_p.Range,sys_p.k_min,ms,sys_p.k_min,'filled');
set(gca,'fontsize',axis_size)
colormap(turbo)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.String = 'K_{min} (N/m)';
title("PL SLM Performance Space",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("Range (m)",'fontSize',label_size)
set(gca,'fontname',font_type)

subplot(2,3,2); hold on
scatter3(sys_p.Weight(c1),sys_p.Range(c1),sys_p.k_min(c1),ms,c1_color,'filled','DisplayName','Class I');
scatter3(sys_p.Weight(c2),sys_p.Range(c2),sys_p.k_min(c2),ms,c2_color,'filled','DisplayName','Class II');
scatter3(sys_p.Weight(c3),sys_p.Range(c3),sys_p.k_min(c3),ms,c3_color,'filled','DisplayName','Class III');
set(gca,'fontsize',axis_size)
title("Class Based Clustering",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("Range (m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

subplot(2,3,3); hold on
scatter3(sys_p.Weight(t1),sys_p.Range(t1),sys_p.k_min(t1),ms,t1_color,'filled','DisplayName','Type I');
scatter3(sys_p.Weight(t2),sys_p.Range(t2),sys_p.k_min(t2),ms,t2_color,'filled','DisplayName','Type II');
set(gca,'fontsize',axis_size)
title("Type Based Clustering",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("Range (m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)


subplot(2,3,4); hold on; grid on;
s = scatter3(sys_p.Weight,sys_p.Range,sys_p.k_min,ms,sys_p.k_min,'filled');
set(gca,'fontsize',axis_size)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.String = 'K_{min} (N/m)';
title("PL SLM Performance Space",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("Range (m)",'fontSize',label_size)
zlabel("K_{min} (N/m)",'fontSize',label_size)
set(gca,'fontname',font_type)

subplot(2,3,5); hold on; grid on;
scatter(sys_p.Range,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p.Range(idxs_rk),sys_p.k_min(idxs_rk),'filled','r','DisplayName',"Pareto Optimal")
set(gca,'fontsize',axis_size)
title("Range vs K_{min} (Pareto Optimal)",'fontSize',title_size)
xlabel("Range (m)",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)


subplot(2,3,6); hold on; grid on;
scatter(sys_p.Weight,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p.Weight(idxs_wk),sys_p.k_min(idxs_wk),'filled','r','DisplayName',"Pareto Optimal")
set(gca,'fontsize',axis_size)
title("Weight vs K_{min} (Pareto Optimal)",'fontSize',title_size)
xlabel("Weight ",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

