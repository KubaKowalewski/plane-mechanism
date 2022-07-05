%% Import Data

close all; clear all; clc;

sys_Rw = readtable("Rw_limits");
sys_Rh = readtable("Rh_limits");
sys_Rr = readtable("Rr_limits");


%% Plotting data
figure(1); clf

% Plotting params
lw = 4;
sw = 4;
label_size = 30;
legend_size = 18;
title_size = 24;
axis_size = 18;
K_color = [0 0.4470 0.7410];
R_color = [0.4660 0.6740 0.1880];
M_color = [0.6350 0.0780 0.1840];

% Normalize wrt to La
Rw_norm = sys_Rw.Rw./sys_Rw.A;
Rh_norm = sys_Rh.Rh./sys_Rh.A;

% Class bounds
c1 = Rw_norm < -1;
c2 = Rw_norm > -1 & Rw_norm < 0;
c3 = Rw_norm > 0;

% Stiffness Plots
subplot(3,3,1); hold on; grid on;
plot(Rh_norm,sys_Rh.k_min,'lineWidth',lw,'Color',K_color)
set(gca,'fontsize',axis_size)
ylim([0 10e7])
ylabel("$K_{min}$ (N/m)",'fontSize',label_size,'interpreter','latex')

subplot(3,3,2); hold on; grid on;
plot(Rw_norm(c1),sys_Rw.k_min(c1),'lineWidth',lw,'Color',K_color,'HandleVisibility','off')
plot(Rw_norm(c2),sys_Rw.k_min(c2),'lineWidth',lw,'Color',K_color,'HandleVisibility','off')
plot(Rw_norm(c3),sys_Rw.k_min(c3),'lineWidth',lw,'Color',K_color,'HandleVisibility','off')
set(gca,'fontsize',axis_size)
xline(0,'r-.','lineWidth',sw)
xline(-1,'r-.','lineWidth',sw)
lg = legend("Singularity",'interpreter','latex');
lg.FontSize = legend_size;
set(lg,'Box','off')
ylim([0 10e7])

subplot(3,3,3); hold on; grid on;
plot(sys_Rr.Rr,sys_Rr.k_min,'lineWidth',lw,'Color',K_color)
set(gca,'fontsize',axis_size)
ylim([0 10e7])

% Range Plots
subplot(3,3,4); hold on; grid on;
plot(Rh_norm,sys_Rh.Range,'lineWidth',lw,'Color',R_color)
set(gca,'fontsize',axis_size)
ylim([0 1])
ylabel("$R$ (m)",'fontSize',label_size,'interpreter','latex')

subplot(3,3,5); hold on; grid on;
plot(Rw_norm(c1),sys_Rw.Range(c1),'lineWidth',lw,'Color',R_color)
plot(Rw_norm(c2),sys_Rw.Range(c2),'lineWidth',lw,'Color',R_color)
plot(Rw_norm(c3),sys_Rw.Range(c3),'lineWidth',lw,'Color',R_color)
set(gca,'fontsize',axis_size)
xline(0,'r-.','lineWidth',sw,'HandleVisibility','off')
xline(-1,'r-.','lineWidth',sw)
ylim([0 1])

subplot(3,3,6); hold on; grid on;
plot(sys_Rr.Rr,sys_Rr.Range,'lineWidth',lw,'Color',R_color)
set(gca,'fontsize',axis_size)
ylim([0 1])

% Mass Plots
subplot(3,3,7); hold on; grid on;
plot(Rh_norm,sys_Rh.Mass,'lineWidth',lw,'Color',M_color)
set(gca,'fontsize',axis_size)
ylabel("$W$ (kg)",'fontSize',label_size,'interpreter','latex')
xlabel("$\overline{R}_{H1}$",'fontSize',label_size,'interpreter','latex')
ylim([0 2])

subplot(3,3,8); hold on; grid on;
plot(Rw_norm(c1),sys_Rw.Mass(c1),'lineWidth',lw,'Color',M_color)
plot(Rw_norm(c2),sys_Rw.Mass(c2),'lineWidth',lw,'Color',M_color)
plot(Rw_norm(c3),sys_Rw.Mass(c3),'lineWidth',lw,'Color',M_color)
set(gca,'fontsize',axis_size)
xline(0,'r-.','lineWidth',sw)
xline(-1,'r-.','lineWidth',sw)
xlabel("$\overline{R}_W$",'fontSize',label_size,'interpreter','latex')
ylim([0 2])

subplot(3,3,9); hold on; grid on;
plot(sys_Rr.Rr,sys_Rr.Mass,'lineWidth',lw,'Color',M_color)
set(gca,'fontsize',axis_size)
xlabel("$R_{Ratio}$",'fontSize',label_size,'interpreter','latex')
ylim([0 2])
