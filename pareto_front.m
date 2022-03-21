%% min stiffness

close all; clear all; clc;
sys_p = readtable("pareto_front");

% apply scaling factors
SD = 1;
SL = 10;
% scale for length
sys_p.L = sys_p.L .* SL;
sys_p.k_max = sys_p.k_max .* 1/SL;
sys_p.k_min = sys_p.k_min .* 1/SL;
sys_p.W = sys_p.W .* SL;
% scale for diameter
sys_p.k_max = sys_p.k_max .* SD^2;
sys_p.k_min = sys_p.k_min .* SD^2;
sys_p.W = sys_p.W .* SD^2;

% Looking at max stiffness of mechanism
figure(1); clf
s = scatter3(sys_p.L,sys_p.W,sys_p.k_min,ones(size(sys_p.k_min)),sys_p.k_min,'filled');
s.SizeData = 50;
s.ButtonDownFcn = @showZValueFcn;
xlabel("Range (m)")
ylabel("Weight (kg)")
zlabel(" Min Stiffness (N/m)")
title("SLM System Property Sweep (K min)")
colorbar
set(gca,'fontsize',16)

% Looking at minumum stiffness of mechanism
figure(2); clf;
s = scatter3(sys_p.L,sys_p.W,sys_p.k_max,ones(size(sys_p.k_max)),sys_p.k_max,'filled');
s.SizeData = 50;
s.ButtonDownFcn = @showZValueFcn;
xlabel("Range (m)")
ylabel("Weight (kg)")
zlabel("Max Stiffness (N/m)")
title("SLM System Property Sweep (K max)")
colorbar
set(gca,'fontsize',16)

% Weighted stiffness of mechanism
Wmax = .5;
Wmin = .5;
k_weighted = Wmax.*sys_p.k_max + Wmin.*sys_p.k_min;

figure(3); clf;
s = scatter3(sys_p.L,sys_p.W,k_weighted,ones(size(sys_p.k_min)),k_weighted,'filled');
s.SizeData = 50;
s.ButtonDownFcn = @showZValueFcn;
xlabel("Range (m)")
ylabel("Weight (kg)")
zlabel(" Weighted Stiffness (N/m)")
title("SLM System Property Sweep (K Weighted)")
colorbar
set(gca,'fontsize',16)

% Find best SLMs
idx = find(sys_p.k_max==max(sys_p.k_max));
fprintf('(Best Max Stiffness) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))

idx = find(sys_p.k_min==max(sys_p.k_min));
fprintf('(Best Min Stiffness) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))

idx = find((sys_p.L./sys_p.W)==max(sys_p.L./sys_p.W));
fprintf('(Best Range/Weight Ratio) A B C = [%0.5f %0.5f %0.5f]\n', sys_p.A(idx), sys_p.B(idx), sys_p.C(idx))


% axh.ButtonDownFcn = {@showZValueFcn, x, y, z};  % old version of answer
function [coordinateSelected, minIdx] = showZValueFcn(hObj, event)
%  FIND NEAREST (X,Y,Z) COORDINATE TO MOUSE CLICK
% Inputs:
%  hObj (unused) the axes
%  event: info about mouse click
% OUTPUT
%  coordinateSelected: the (x,y,z) coordinate you selected
%  minIDx: The index of your inputs that match coordinateSelected. 
x = hObj.XData; 
y = hObj.YData; 
z = hObj.ZData; 
pt = event.IntersectionPoint;       % The (x0,y0,z0) coordinate you just selected
coordinates = [x(:),y(:),z(:)];     % matrix of your input coordinates
dist = pdist2(pt,coordinates);      %distance between your selection and all points
[~, minIdx] = min(dist);            % index of minimum distance to points
coordinateSelected = coordinates(minIdx,:); %the selected coordinate
% from here you can do anything you want with the output.  This demo
% just displays it in the command window.  
fprintf('Point Index = %d\n', minIdx)
end % <--- optional if this is embedded into a function
