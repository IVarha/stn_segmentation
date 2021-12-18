clear;
close all;
load("Z:\processing_data\mah_dist_electrodes.mat")
mesh_f1 = "Z:\processing_data\new_data_sorted\sub-P"+ left_name+"\3_1T1.obj";
mesh_f2 = "Z:\processing_data\new_data_sorted\sub-P"+ right_name +"\4_1T1.obj" ;


[f1,v1] = read_obj(mesh_f1);
c = transpose(10./left);
%c(224) = 12;
%c = transpose(left);
triang = triangulation(f1,v1);
trisurf(triang,'FaceVertexCData',c,'FaceAlpha',1,'LineWidth',0.2)
colormap('jet')
%axis equal;
title("Left")
hold on;
colorbar;
%scatter3(right_position(:,1),right_position(:,2),right_position(:,3),500,'magenta','filled');
%line 3d
%view([-37.5 -30])

n_line_left = left_position(1,:) - left_position(2,:);

p_r = left_position(1,:);
X_r = p_r(1) + n_line_left(1)*(-1.5:0.1:1);
Y_r = p_r(2) + n_line_left(2)*(-1.5:0.1:1);
Z_r = p_r(3) + n_line_left(3)*(-1.5:0.1:1);
%r_line = plot3(X_r,Y_r,Z_r);
%r_line.LineWidth = 10;
%r_line.Color = "magenta";
axis equal;
%scatter3(left_pts_in_out(:,1,1),left_pts_in_out(:,1,2),left_pts_in_out(:,1,3),50,'yellow','filled');
%scatter3(left_pts_in_out(:,2,1),left_pts_in_out(:,2,2),left_pts_in_out(:,2,3),50,'c','filled');




figure;
[f5,v5] = read_obj(mesh_f2);

triang = triangulation(f5,v5);
c2 = transpose(10./right);
%c2(144)=13;
trisurf(triang,'FaceVertexCData',c2,'FaceAlpha',1,'LineWidth',0.2)
colormap('jet')
axis equal;
title("Right")
%view([30 0])
colorbar;
hold on;
%scatter3(left_position(:,1),left_position(:,2),left_position(:,3),500,'magenta','filled');
%n_line_right = right_position(1,:) - right_position(2,:);

p_l = right_position(1,:);
%X_l = p_l(1) + n_line_right(1)*(-1.5:0.1:1);
%Y_l = p_l(2) + n_line_right(2)*(-1.5:0.1:1);
%Z_l = p_l(3) + n_line_right(3)*(-1.5:0.1:1);
%l_line = plot3(X_l,Y_l,Z_l);
%l_line.LineWidth = 10;
%l_line.Color = "magenta";
