close all; clear all; clc;
sys_p = readtable("sim_results3");

v1 = sys_p.Rh>0;
v2 = sys_p.Rh<0 & sys_p.Rh>-sys_p.A;
v3 = sys_p.Rh<-sys_p.A;

t1 = ismember(sys_p.Rr,1) & ismember(sys_p.Rf,"False");
t2 = sys_p.Rr < 1 & ismember(sys_p.Rf,"False");
t3 = sys_p.Rr < 1 & ismember(sys_p.Rf,"True");

alpha = .7;
fs = 18;

subplot(2,3,[1,4]); hold on; grid on
s = scatter3(sys_p.Range,sys_p.Weight,sys_p.k_min,ones(size(sys_p.k_min)),sys_p.k_min,'filled');
s.SizeData = 50;


s2 = subplot(2,3,2); hold on
scatter(sys_p.Range(v1),sys_p.k_min(v1),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Range(v2),sys_p.k_min(v2),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Range(v3),sys_p.k_min(v3),'filled','MarkerFaceAlpha',alpha)
legend("Class I","Class II","Class III","FontSize",fs)
xlabel("Range (m)","FontSize",fs)
ylabel("Stiffness (N/m)","FontSize",fs)

s3 = subplot(2,3,3); hold on
scatter(sys_p.Weight(v1),sys_p.k_min(v1),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Weight(v2),sys_p.k_min(v2),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Weight(v3),sys_p.k_min(v3),'filled','MarkerFaceAlpha',alpha)
legend("Class I","Class II","Class III","FontSize",fs)
xlabel("Weight (kg)","FontSize",fs)
ylabel("Stiffness (N/m)","FontSize",fs)


s4 = subplot(2,3,5); hold on
scatter(sys_p.Range(t2),sys_p.k_min(t2),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Range(t1),sys_p.k_min(t1),'filled','MarkerFaceAlpha',alpha) 
legend("Type Y","Type X","FontSize",fs)
xlabel("Range (m)","FontSize",fs)
ylabel("Stiffness (N/m)","FontSize",fs)

s5 = subplot(2,3,6); hold on
scatter(sys_p.Weight(t2),sys_p.k_min(t2),'filled','MarkerFaceAlpha',alpha)
scatter(sys_p.Weight(t1),sys_p.k_min(t1),'filled','MarkerFaceAlpha',alpha)
legend("Type Y","Type X","FontSize",fs)
xlabel("Weight (kg)","FontSize",fs)
ylabel("Stiffness (N/m)","FontSize",fs)


newcolors = [0.4940 0.1840 0.5560
             0.4660 0.6740 0.1880
             0.3010 0.7450 0.9330];


colororder(s4,newcolors)
colororder(s5,newcolors)

%%

figure(3)
scatter(sys_p.Range,sys_p.Range.*sys_p.k_min)
figure(4)
scatter(sys_p.Weight,sys_p.Weight.*sys_p.k_min)