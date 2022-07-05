%% Import Data

close all; clear all; clc;
sys_p = readtable("sim_results3");

%% Pareto search

% Find pareto optimal Range-Kmin by Class
pts = [sys_p.Range,sys_p.k_min];
[ ~, idxs_rk] = paretoFront(pts);

% Find pareto optimal Weight-Kmin by Class
pts = [1./sys_p.Weight,sys_p.k_min];
[ ~, idxs_wk] = paretoFront(pts);

sys_p_rk = sys_p(idxs_rk,:);
sys_p_wk = sys_p(idxs_wk,:);

%% Sorting

 
% Range-KMin Class Sorting
c1_rk = sys_p_rk.Rh./sys_p_rk.A > 0;
c2_rk = sys_p_rk.Rh./sys_p_rk.A < 0 & sys_p_rk.Rh./sys_p_rk.A > -1;
c3_rk = sys_p_rk.Rh./sys_p_rk.A < -1;

% Weight-KMin Class Sorting
c1_wk = sys_p_wk.Rh./sys_p_wk.A > 0;
c2_wk = sys_p_wk.Rh./sys_p_wk.A < 0 & sys_p_wk.Rh./sys_p_wk.A > -1;
c3_wk = sys_p_wk.Rh./sys_p_wk.A < -1;
c1_color = [0 0.4470 0.7410];
c2_color = [0.8500 0.3250 0.0980];
c3_color = [0.9290 0.6940 0.1250];


% Range-KMin Type Sorting
t1_rk = ismember(sys_p_rk.Rr,1) & ismember(sys_p_rk.Rf,"False");
t2_rk = sys_p_rk.Rr < 1 & ismember(sys_p_rk.Rf,"False");

% Weight-KMin Type Sorting
t1_wk = ismember(sys_p_wk.Rr,1) & ismember(sys_p_wk.Rf,"False");
t2_wk = sys_p_wk.Rr < 1 & ismember(sys_p_wk.Rf,"False");

t1_color = [0.4940 0.1840 0.5560];
t2_color = [0.4660 0.6740 0.1880];


%% Data Visualization

close all;

% Optimal Curves
v_color = [.7 .7 .7];

% Plotting parameters
ms = 50;
label_size = 20;
legend_size = 18;
title_size = 24;
axis_size = 16;
font_type = 'arial';

figure(1); clf

subplot(2,4,[1,2,5,6]); hold on; grid on
s = scatter3(sys_p.Weight,sys_p.Range,sys_p.k_min,ms,sys_p.k_min,'filled');
set(gca,'fontsize',axis_size)
colormap(turbo)
cb = colorbar;
set(cb,'FontSize',label_size)
cb.Label.String = 'K_{min} (N/m)';
title("PL SLM Performance Space",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("Range (m)",'fontSize',label_size)
zlabel("K_{min} (N/m)")
set(gca,'fontname',font_type)

subplot(2,4,3); hold on;
scatter(sys_p.Range,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p_rk.Range(c1_rk),sys_p_rk.k_min(c1_rk),ms,c1_color,'filled','DisplayName',"Class I")
scatter(sys_p_rk.Range(c2_rk),sys_p_rk.k_min(c2_rk),ms,c2_color,'filled','DisplayName',"Class II")
scatter(sys_p_rk.Range(c3_rk),sys_p_rk.k_min(c3_rk),ms,c3_color,'filled','DisplayName',"Class III")
set(gca,'fontsize',axis_size)
title("Range vs K_{min}",'fontSize',title_size)
xlabel("Range (m)",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

subplot(2,4,4); hold on;
scatter(sys_p.Range,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p_rk.Range(t1_rk),sys_p_rk.k_min(t1_rk),ms,t1_color,'filled','DisplayName',"Type X")
scatter(sys_p_rk.Range(t2_rk),sys_p_rk.k_min(t2_rk),ms,t2_color,'filled','DisplayName',"Type Y")
set(gca,'fontsize',axis_size)
title("Range vs K_{min}",'fontSize',title_size)
xlabel("Range (m)",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

subplot(2,4,7); hold on;
scatter(sys_p.Weight,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p_wk.Weight(c1_wk),sys_p_wk.k_min(c1_wk),ms,c1_color,'filled','DisplayName',"Class I")
scatter(sys_p_wk.Weight(c2_wk),sys_p_wk.k_min(c2_wk),ms,c2_color,'filled','DisplayName',"Class II")
scatter(sys_p_wk.Weight(c3_wk),sys_p_wk.k_min(c3_wk),ms,c3_color,'filled','DisplayName',"Class III")
set(gca,'fontsize',axis_size)
title("Weight vs K_{min}",'fontSize',title_size)
xlabel("Weight (kg)",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

subplot(2,4,8); hold on;
scatter(sys_p.Weight,sys_p.k_min,ms,v_color,'filled','HandleVisibility','off')
scatter(sys_p_wk.Weight(t1_wk),sys_p_wk.k_min(t1_wk),ms,t1_color,'filled','DisplayName',"Type X")
scatter(sys_p_wk.Weight(t2_wk),sys_p_wk.k_min(t2_wk),ms,t2_color,'filled','DisplayName',"Type Y")
set(gca,'fontsize',axis_size)
title("Weight vs K_{min}",'fontSize',title_size)
xlabel("Weight ",'fontSize',label_size)
ylabel("K_{min} (N/m)",'fontSize',label_size)
lg = legend();
lg.FontSize = legend_size;
set(lg,'Box','off')
set(gca,'fontname',font_type)

