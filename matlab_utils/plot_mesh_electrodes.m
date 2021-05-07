clear;
close all;
load("Z:\processing_data\mah_dist_electrodes.mat")
mesh_f1 = "Z:\processing_data\new_data_sorted\sub-P"+ right_name+"\3_1T1.obj";
mesh_f2 = "Z:\processing_data\new_data_sorted\sub-P"+ left_name +"\4_1T1.obj" ;


[f1,v1] = read_obj(mesh_f1);
c = transpose(10./right);
%c = transpose(left);
triang = triangulation(f1,v1);
trisurf(triang,'FaceVertexCData',c,'FaceAlpha',1,'LineWidth',0.2)
colormap('jet')
%axis equal;
title("Right")
hold on;
colorbar;
scatter3(right_position(:,1),right_position(:,2),right_position(:,3),500,'magenta','filled');
axis equal;

figure;
title("Reft")
[f1,v1] = read_obj(mesh_f2);

triang = triangulation(f1,v1);
c2 = transpose(10./left);
trisurf(triang,'FaceVertexCData',c2,'FaceAlpha',1,'LineWidth',0.2)
colormap('jet')
axis equal;
title("Left")
colorbar;
hold on;
scatter3(left_position(:,1),left_position(:,2),left_position(:,3),500,'magenta','filled');
