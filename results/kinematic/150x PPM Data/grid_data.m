%% Data intake and fitting

% Pull in data
close all; clear; clc;
T = readtable('./7-20-2022-Trial_1.csv');
T = rmmissing(T);
T.X = T.X;
T.Y = T.Y;
T.Z = T.Z/1000; % Convert to in

% Specify dimensions of grid
N = 9;
M = 9;

% Find best fit plane
[n,V,p] = plane_fit([T.X,T.Y,T.Z]);
a = n(1);
b = n(2);
c = n(3);
d = a*p(1)+b*p(2)+c*p(3);

% Set limits for fit plane
n = 2;
pad = 1.1;
x = linspace(pad*min(T.X),pad*max(T.X),2);
y = linspace(pad* min(T.Y),pad*max(T.Y),2);
[X_fit, Y_fit] = meshgrid(x,y);

% Find the distance from the plane to each point
D = abs(a.*T.X+b.*T.Y+c.*T.Z-d)./sqrt(a^2+b^2+c^2);


%% Plot of all points

figure(1); clf; hold on;

axis([min(T.X),max(T.X),min(T.Y),max(T.Y),min(T.Z),max(T.Z)])
grid on
title("PPM Surface Analysis")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (in)");
ylabel("Y (in)");
zlabel("Z (in)");

% Plot all points
scatter3(T.X,T.Y,T.Z,'filled');

%% Create plot of best fit plane
figure(2); clf; hold on
    
axis([min(T.X),max(T.X),min(T.Y),max(T.Y),min(T.Z),max(T.Z)*2])
grid on
title("PPM Grid Surface")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (in)");
ylabel("Y (in)");
zlabel("Z (in)");

% Plot all points
scatter3(T.X,T.Y,T.Z,'filled');

% Plot the plane of best fit
Z_fit = (-(a.*X_fit + b.*Y_fit)+d)./c;
surf(X_fit,Y_fit,Z_fit,'facecolor','blue','facealpha',0.2)

%% Create Colormap plot

figure(3); clf; hold on
z_pad = 5*10^-3;
axis([X_fit(1,1),X_fit(1,end),Y_fit(1,1),Y_fit(end,1),min(T.Z)-z_pad,max(T.Z)+z_pad])
grid on
title("PPM Surface Analysis")
set(gca,'fontsize',16)

% Plot all points
xlabel("X (in)");
ylabel("Y (in)");
zlabel("Z");

% Plot all the points
X = reshape(T.X,[N,M]);
Y = reshape(T.Y,[N,M]);
Z = reshape(T.Z,[N,M]);
D_error = reshape(D,[N,M]);
surf(X,Y,Z,D_error)
hcb = colorbar;
clim([min(D_error,[],'all') max(D_error,[],'all')])
hcb.Label.String = 'Error (in)';
surf(X_fit,Y_fit,Z_fit,'facecolor','black','facealpha',0.1)

%% Create Error Histrogram Distribution

figure(4); clf; hold on
histogram(D)
title("Distribution of Error")
xlabel("Error(in)")
ylabel("Count")