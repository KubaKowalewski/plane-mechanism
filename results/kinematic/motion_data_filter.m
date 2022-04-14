% Pull in data
clear all; clc;
T = readtable('./rolling_1_trim.xlsx');
T = rmmissing(T);
T.X = T.X*1000;
T.Y = T.Y*1000;
T.Z = T.Z*1000;
% Find best fit plane
[n,V,p] = plane_fit([T.X,T.Y,T.Z]);
a= n(1);
b = n(2);
c = n(3);
d = a*p(1)+b*p(2)+c*p(3);

% Set limits for fit plane
n = 2;
x = linspace(min(T.X),max(T.X),n);
y = linspace(min(T.Y),max(T.Y),n);
[X,Y] = meshgrid(x,y);

% Find the distance from the plane to each point
D = abs(a.*T.X+b.*T.Y+c.*T.Z-d)./sqrt(a^2+b^2+c^2);

XX = T.X(D<0.5);
YY = T.Y(D<0.5);
ZZ = T.Z(D<0.5);

% Find best fit plane again
[n,V,p] = plane_fit([XX,YY,ZZ]);
a= n(1);
b = n(2);
c = n(3);
d = a*p(1)+b*p(2)+c*p(3);

% Set limits for fit plane
n = 2;
x = linspace(min(XX),max(XX),n);
y = linspace(min(YY),max(YY),n);
[X,Y] = meshgrid(x,y);

% Find the distance from the plane to each point
D = abs(a.*XX+b.*YY+c.*ZZ-d)./sqrt(a^2+b^2+c^2);

figure(1); clf; hold on;

axis equal
grid on
title("Granite Slab Rolling Test")
set(gca,'fontsize',24)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all points
scatter3(XX,YY,ZZ,'filled');
%% Create plot of best fit plane
figure(2); clf; hold on

axis equal
grid on
title("Granite Slab Rolling Test")
set(gca,'fontsize',24)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all points
scatter3(XX,YY,ZZ,'filled');

% Plot the plane of best fit
Z = (-(a.*X + b.*Y)+d)./c;
surf(X,Y,Z,'facecolor','blue','facealpha',0.2)
%% Create Colormap plot

figure(3); clf; hold on

axis equal
grid on
title("Granite Slab Rolling Test")
set(gca,'fontsize',24)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all the points
scatter3(XX,YY,ZZ,[],D,'filled');
hcb = colorbar;
hcb.Title.String = "Error (mm)";

figure(4); clf; hold on
histogram(D)
title("Error Histrogram")