%% min stiffness

close all; clear all; clc;
sys_p = readtable("sim_results4");

% apply scaling factors
SD = 1;
SL = 1;

% scale for length
sys_p.Range = sys_p.Range .* SL;
sys_p.k_max = sys_p.k_max .* 1/SL;
sys_p.k_min = sys_p.k_min .* 1/SL;
sys_p.Mass = sys_p.Mass .* SL;
sys_p.A = sys_p.A.*SL;

% scale for diameter
sys_p.k_max = sys_p.k_max .* SD^2;
sys_p.k_min = sys_p.k_min .* SD^2;
sys_p.Mass = sys_p.Mass .* SD^2;

% Looking at min stiffness of mechanism
figure(1); clf
s = scatter3(sys_p.Range,sys_p.Mass,sys_p.k_min,ones(size(sys_p.k_min)),sys_p.k_min,'filled');
s.SizeData = 50;
s.ButtonDownFcn = @showZValueFcn;
xlabel("Range (m)")
ylabel("Mass (kg)")
zlabel(" Min Stiffness (N/m)")
title("SLM System Property Sweep (K min)")
colorbar
set(gca,'fontsize',20)

% Looking at max stiffness of mechanism
figure(2); clf;
s = scatter3(sys_p.Range,sys_p.Mass,sys_p.k_max,ones(size(sys_p.k_max)),sys_p.k_max,'filled');
s.SizeData = 50;
s.ButtonDownFcn = @showZValueFcn;
xlabel("Range (m)")
ylabel("Mass (kg)")
zlabel("Max Stiffness (N/m)")
title("SLM System Property Sweep (K max)")
colorbar
set(gca,'fontsize',20)

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
