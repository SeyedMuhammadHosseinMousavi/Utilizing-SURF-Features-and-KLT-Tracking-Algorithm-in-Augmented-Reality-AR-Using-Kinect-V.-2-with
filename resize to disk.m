clc;
clear;;
path='testfolder';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : 20
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
% Resize
for i = 1:20   
sizee{i} = imresize(images{i},[512 512]);
end;
%save to disk
for i = 1:20   
   imwrite(sizee{i},strcat('my_new',num2str(i),'.jpg'));
end