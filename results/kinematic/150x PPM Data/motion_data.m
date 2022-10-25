% Pull in data
clear all; clc;
T = readtable('./flexure_6-2-2022-export.csv');
T = rmmissing(T);
T.X = T.X;
T.Y = T.Y;
T.Z = T.Z;
% Find best fit plane
[n,V,p] = plane_fit([T.X,T.Y,T.Z]);
a= n(1);
b = n(2);
c = n(3);
d = a*p(1)+b*p(2)+c*p(3);

% Set limits for fit plane
n = 2;
x = linspace(min(T.X),max(T.X),2);
y = linspace(min(T.Y),max(T.Y),2);
[X,Y] = meshgrid(x,y);

% Find the distance from the plane to each point
D = abs(a.*T.X+b.*T.Y+c.*T.Z-d)./sqrt(a^2+b^2+c^2);


%% Plot of just points

figure(1); clf; hold on;

axis equal
grid on
title("PL-PPM Opitrack Motion")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all points
scatter3(T.X,T.Y,T.Z,'filled');

%% Create plot of best fit plane
figure(2); clf; hold on
    
axis equal
grid on
title("PL-PPM Opitrack Motion")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all points
scatter3(T.X,T.Y,T.Z,'filled');

% Plot the plane of best fit
Z = (-(a.*X + b.*Y)+d)./c;
surf(X,Y,Z,'facecolor','blue','facealpha',0.2)

%% Create Colormap plot

figure(3); clf; hold on

axis equal
grid on
title("PL-PPM Error Analysis")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");

% Plot all the points
scatter3(T.X,T.Y,T.Z,[],D,'filled');
hcb = colorbar;
hcb.Label.String = 'Error (mm)';

%% Create Error Histrogram Distribution
figure(4); clf; hold on
histogram(D)
title("Distribution of Error")
xlabel("Error(mm)")
ylabel("Count")