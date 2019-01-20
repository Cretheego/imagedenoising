clear all
clc
clf
%colormap(gray)

%maindir = uigetdir( '选择一个文件夹' );
sigma1 = 10;  %标准差
sigma = (sigma1/255)^2;%0.25  0.16  0.09 0.04 0.01
maindir1 = 'D:\\jupyter_notebook\\cifar\\test';
maindir = 'D:\\jupyter_notebook\\cifar\\train';
%0.2表示speckle噪声，0.1表示salt & pepper噪声，0.3表示高斯噪声
targetdir_clean1 = ['D:\\jupyter_notebook\\cifar\\compare\\test\\clean_test',num2str(sigma1+0.3)];
targetdir_clean = ['D:\\jupyter_notebook\\cifar\\compare\\train\\clean_train',num2str(sigma1+0.3)];
targetdir_noisy1 = ['D:\\jupyter_notebook\\cifar\\compare\\test\\noisy_test',num2str(sigma1+0.3)];
targetdir_noisy = ['D:\\jupyter_notebook\\cifar\\compare\\train\\noisy_train',num2str(sigma1+0.3)];

mkdir(targetdir_clean1);
mkdir(targetdir_clean);
mkdir(targetdir_noisy1);
mkdir(targetdir_noisy);

filelist1 = dir(maindir1);
filelist  = dir(maindir);
fileNum1 = size(filelist1,1); 
fileNum = size(filelist,1); 

if 1
    for i = 3 : fileNum
        file_name = fullfile(maindir, filelist(i).name);
        img = double(imread(file_name))/255;
        ima = (rgb2gray((img)));
        imwrite(ima,fullfile(targetdir_clean, filelist(i).name));
        rima = imnoise((rgb2gray(img)),'gaussian',0,sigma);
        %rima = imnoise((rgb2gray(img)),'speckle',sigma);
        %rima = imnoise((rgb2gray(img)),'salt & pepper',sigma);
        imwrite(rima,fullfile(targetdir_noisy, filelist(i).name));
        i
    end
end

for i = 3 : fileNum1
    file_name1 = fullfile(maindir1, filelist1(i).name);
    img1 = double(imread(file_name1))/255;
    ima1 = (rgb2gray((img1)));
    %fullfile(targetdir_clean1, filelist1(i).name);
    imwrite(ima1,fullfile(targetdir_clean1, filelist1(i).name));
    rima1 = imnoise(ima1,'gaussian',0,sigma);
    %rima1 = imnoise(ima1,'speckle',sigma);
    %rima1 = imnoise(ima1,'salt & pepper',sigma);
    imwrite(rima1,fullfile(targetdir_noisy1, filelist1(i).name));
    i
end
%img = imread(fullfile(maindir1, filelist1(3).name))
%ima1 = (rgb2gray((img1)))
%save(fullfile(targetdir, '\noisy_train.mat'),'rima');
%save(fullfile(targetdir, '\clean_train.mat'),'ima');