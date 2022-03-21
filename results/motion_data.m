% Pull in data
T = readtable('opti-3-17_steel.csv');
figure(1); clf; hold on

% Plot all points
scatter3(T.X,T.Y,T.Z,'.');
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Find best fit plane
[n,V,p] = plane_fit([T.X,T.Y,T.Z]);


% plot the two adjusted planes
% [X,Y] = meshgrid([max(T.Y), min(T.Y)],[max(T.Z), min(T.Z)]);
% 
% first plane
% surf(X,Y, - (n_1(1)/n_1(3)*X+n_1(2)/n_1(3)*Y-dot(n_1,p_1)/n_1(3)),'facecolor','red','facealpha',0.5);

% axis equal
grid on
t = linspace(0,2*pi);
plot(sin(t),2*cos(t))
grid on
pbaspect([1 1 1])
ylim([190 300])
title("Steel PPM Path")
set(gca,'fontsize',16)
%%
T = readtable('PPM_MAG_TEST1.csv');
figure(2); clf;
scatter3(T.X,T.Y,T.Z,'r.');
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");
title("Wood PPM Path")
axis equal
set(gca,'fontsize',16)

%%