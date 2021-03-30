clear;
mesh_f1 = "Z:\processing_data\workdir\3_mean.obj";
mesh_f2 = "Z:\processing_data\workdir\4_mean.obj";
clust_f1 = "Z:\processing_data\workdir\2clusters.txt";

colors = ['r','g','b','m','y','c'];

cf1 = load(clust_f1);
st1 = unique(cf1); 
[f1,v1] = read_obj(mesh_f1);

triang = triangulation(f1,v1);
trisurf(triang,'FaceColor','b','FaceAlpha',0,'LineWidth',0.2)

hold on;
for i=1:length(st1)
    pts = v1(cf1==st1(i),:);
    scatter3(pts(:,1),pts(:,2),pts(:,3),'filled','MarkerFaceColor',colors(i));
  
end
axis square;
%figure;
clust_f2 = "Z:\processing_data\workdir\3clusters.txt";
cf2 = load(clust_f2);
st2 = unique(cf2); 
[f1,v1] = read_obj(mesh_f2);

triang = triangulation(f1,v1);
trisurf(triang,'FaceColor','b','FaceAlpha',0,'LineWidth',0.2)

hold on;
for i=1:length(st2)
    pts = v1(cf2==st2(i),:);
    scatter3(pts(:,1),pts(:,2),pts(:,3),'filled','MarkerFaceColor',colors(i));
  
end