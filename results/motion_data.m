T = readtable('PPM_TEST1.csv');
figure(1); clf;
scatter3(T.X,T.Y,T.Z,'.');
xlabel("X (mm)");
ylabel("Y (mm)");
zlabel("Z (mm)");
axis equal
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