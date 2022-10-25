%% Plotting Parameters
close all; clear; clc;
AZ = 120;
EL = 20;
z_pad = 2*10^-3;


%% Data intake and fitting for surface 1

% Pull in data for surface1
T1 = readtable('./110x PPM Data/7-21-2022-Trial_1.csv');
T1 = rmmissing(T1);
T1.X = T1.X;
T1.Y = T1.Y;
T1.Z = T1.Z/1000; % Convert to in

% Specify dimensions of grid
N = 9;
M = 9;

% Find best fit plane
[n1,V1,p1] = plane_fit([T1.X,T1.Y,T1.Z]);
a1 = n1(1);
b1 = n1(2);
c1 = n1(3);
d1 = a1*p1(1)+b1*p1(2)+c1*p1(3);

% Set limits for fit plane
n1 = 2;
pad = 1.1;
x1 = linspace(pad*min(T1.X),pad*max(T1.X),2);
y1 = linspace(pad* min(T1.Y),pad*max(T1.Y),2);
[X1_fit, Y1_fit] = meshgrid(x1,y1);

% Find the distance from the plane to each point
D1 = abs(a1.*T1.X+b1.*T1.Y+c1.*T1.Z-d1)./sqrt(a1^2+b1^2+c1^2);
Z1_fit = (-(a1.*X1_fit + b1.*Y1_fit)+d1)./c1;

%% Data intake and fitting for surface 2

% Pull in data for surface1
T2 = readtable('./150x PPM Data/7-20-2022_Average.csv');
T2 = rmmissing(T2);
T2.X = T2.X;
T2.Y = T2.Y;
T2.Z = T2.Z/1000; % Convert to in

% Specify dimensions of grid
N = 9;
M = 9;

% Find best fit plane
[n2,V2,p2] = plane_fit([T2.X,T2.Y,T2.Z]);
a2 = n2(1);
b2 = n2(2);
c2 = n2(3);
d2 = a2*p2(1)+b2*p2(2)+c2*p2(3);

% Set limits for fit plane
n2 = 2;
pad = 1.1;
x2 = linspace(pad*min(T2.X),pad*max(T2.X),2);
y2 = linspace(pad* min(T2.Y),pad*max(T2.Y),2);
[X2_fit, Y2_fit] = meshgrid(x2,y2);

% Find the distance from the plane to each point
D2 = abs(a2.*T2.X+b2.*T2.Y+c2.*T2.Z-d2)./sqrt(a2^2+b2^2+c2^2);
Z2_fit = (-(a2.*X2_fit + b2.*Y2_fit)+d2)./c2;


%% Create Colormap plot for surface 1

figure(2); clf;
subplot(1,2,1); hold on

axis([X1_fit(1,1),X1_fit(1,end),Y1_fit(1,1),Y1_fit(end,1),min(T1.Z)-z_pad,max(T1.Z)+z_pad])

% Plot all the points
X1 = reshape(T1.X,[N,M]);
Y1 = reshape(T1.Y,[N,M]);
Z1 = reshape(T1.Z,[N,M]);
D1_error = reshape(D1,[N,M]);
Z1_interp = (-(a1.*X1 + b1.*Y1)+d1)./c1;
RMSE = sqrt(mean((Z1_interp-Z1).^2,'all'));

% Format plot
grid on
title(sprintf('Planarity Analysis (RMSE:%0.4f)',RMSE))
set(gca,'fontsize',16)
xlabel("X (in)");
ylabel("Y (in)");
zlabel("Z");

% Plot surface 1
surf(X1,Y1,Z1,D1_error,'FaceColor',"interp")
colormap turbo	
hcb = colorbar;
clim([min([D1,D2],[],"All"),max([D1,D2],[],"All")])
hcb.Label.String = 'Error (in)';
surf(X1_fit,Y1_fit,Z1_fit,'facecolor','black','facealpha',0.1)
view([AZ EL])

%%
subplot(1,2,2); hold on

axis([X2_fit(1,1),X2_fit(1,end),Y2_fit(1,1),Y2_fit(end,1),min(T2.Z)-z_pad,max(T2.Z)+z_pad])

% Plot all the points
X2 = reshape(T2.X,[N,M]);
Y2 = reshape(T2.Y,[N,M]);
Z2 = reshape(T2.Z,[N,M]);
D2_error = reshape(D2,[N,M]);
Z2_interp = (-(a2.*X2 + b2.*Y2)+d2)./c2;
RMSE = sqrt(mean((Z2_interp-Z2).^2,'all'));

% Format plot
grid on
title(sprintf('Planarity Analysis (RMSE:%0.4f)',RMSE))
set(gca,'fontsize',16)
xlabel("X (in)");
ylabel("Y (in)");
zlabel("Z");

% Plot surface 1
surf(X2,Y2,Z2,D2_error,'FaceColor',"interp")
colormap turbo	
hcb = colorbar;
clim([min([D1,D2],[],"All"),max([D1,D2],[],"All")])
hcb.Label.String = 'Error (in)';
surf(X2_fit,Y2_fit,Z2_fit,'facecolor','black','facealpha',0.1)
view([AZ EL])