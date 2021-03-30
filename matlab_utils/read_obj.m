function [faces,vert] = read_obj(filename)
%READ_OBJ Summary of this function goes here
%   Detailed explanation goes here
fileID = fopen(filename,'r');
all_chars = fscanf(fileID,"%c");
newStr = split(all_chars);
ind = 1;
endl = 0;

vert = [];
while ind < length(newStr)
    a = newStr(ind);
    a = a{1};
    if a=='v'
        pt = [0,0,0];
        ind = ind +1;
        a = newStr(ind);
        pt(1) = str2double(a{1});
        
        ind = ind +1;
        a = newStr(ind);
        pt(2) = str2double(a{1});
        
        ind = ind +1;
        a = newStr(ind);
        pt(3) = str2double(a{1});
        vert = [vert;pt];
    end
    ind = ind +1;
end
faces = [];
ind = 1;
while ind < length(newStr)
    a = newStr(ind);
    a = a{1};
    if isempty(a)
        ind = ind +1;
        continue;
    end
    if a=='f'
        pt = [0,0,0];
        ind = ind +1;
        a = newStr(ind);
        pt(1) = str2double(a{1});
        
        ind = ind +1;
        a = newStr(ind);
        pt(2) = str2double(a{1});
        
        ind = ind +1;
        a = newStr(ind);
        pt(3) = str2double(a{1});
        faces = [faces;pt];
    end
    ind = ind +1;
end

end

