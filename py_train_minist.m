clear all
clc
% 对minist数据集进行加噪处理
%maindir = uigetdir( '选择一个文件夹' );
sigma1 = 90;  %标准差
sigma = 0.25; %0.25  0.16  0.09 0.04 0.01;(sigma1/255)^2
%maindir3 = 'K:\\imagenet\\originalPics\\1_gray';
maindir2 = 'D:\\jupyter_notebook\\minist\\clean\\val';
maindir1 = 'D:\\jupyter_notebook\\minist\\clean\\test';
maindir = 'D:\\jupyter_notebook\\minist\\clean\\train';
%0.2表示speckle噪声，0.1表示salt & pepper噪声，0.3表示高斯噪声
%targetdir_clean1 = ['D:\\jupyter_notebook\\cifar\\compare\\test\\clean_test',num2str(sigma1+0.3)];
%targetdir_clean = ['D:\\jupyter_notebook\\cifar\\compare\\train\\clean_train',num2str(sigma1+0.3)];
%targetdir_noisy3 = ['K:\\imagenet\\originalPics\\1_gray_noise'];
targetdir_noisy2 = ['D:\\jupyter_notebook\\minist\\noisy\\gauss\\val\\noisy_val',num2str(sigma1+0.1)];
targetdir_noisy1 = ['D:\\jupyter_notebook\\minist\\noisy\\gauss\\test\\noisy_test',num2str(sigma1+0.1)];
targetdir_noisy = ['D:\\jupyter_notebook\\minist\\noisy\\gauss\\train\\noisy_train',num2str(sigma1+0.1)];

%mkdir(targetdir_clean1);
%mkdir(targetdir_clean);
mkdir(targetdir_noisy2);
mkdir(targetdir_noisy1);
mkdir(targetdir_noisy);

%filelist3 = dir(maindir3);
filelist2 = dir(maindir2);
filelist1 = dir(maindir1);
filelist  = dir(maindir);
%fileNum3 = size(filelist3,1); 
fileNum2 = size(filelist2,1); 
fileNum1 = size(filelist1,1); 
fileNum = size(filelist,1); 
%训练
if 1
    for i = 3 : fileNum
        file_name = fullfile(maindir, filelist(i).name);
        img = double(imread(file_name))/255;
        if ndims(img)==3 %如果不是灰度图像，转换为灰度
            img = rgb2gray(img);
            imwrite(ima,file_name);
        end
        ima = img;
        %rima = imnoise(((ima)),'gaussian',0,sigma);
        %rima = imnoise((rgb2gray(img)),'gaussian',0,sigma);
        %rima = imnoise((rgb2gray(img)),'speckle',sigma);
        rima = imnoise(((img)),'salt & pepper',sigma);
        imwrite(rima,fullfile(targetdir_noisy, filelist(i).name));
        i
    end
end
%测试集
if 1
    for i = 3 : fileNum1
        file_name1 = fullfile(maindir1, filelist1(i).name);
        img1 = double(imread(file_name1))/255;
        if ndims(img1)==3 %如果不是灰度图像，转换为灰度
            img1 = rgb2gray(img1);
            imwrite(ima1,fullfile(maindir1,[num2str(i-2),'.png']));
        end
        ima1 = img1;
        %imwrite(ima1,fullfile(maindir1,[num2str(i-2),'.png']));
        %rima1 = imnoise(ima1,'gaussian',0,sigma);
        %rima1 = imnoise(ima1,'speckle',sigma);
        rima1 = imnoise(ima1,'salt & pepper',sigma);
        %imwrite(rima1,fullfile(targetdir_noisy1, [num2str(i-2),'.png']));
        imwrite(rima1,fullfile(targetdir_noisy1, filelist1(i).name));
        i
    end
end
%验证
if 1
    for i = 3 : fileNum2
        file_name2 = fullfile(maindir2, filelist2(i).name);
        img2 = double(imread(file_name2))/255;
        if ndims(img2)==3 %如果不是灰度图像，转换为灰度
            img2 = rgb2gray(img2);
            imwrite(ima2,file_name2);
        end
        ima2 = img2;
        %rima2 = imnoise(ima2,'gaussian',0,sigma);
        %rima2 = imnoise(ima1,'speckle',sigma);
        rima2 = imnoise(ima2,'salt & pepper',sigma);
        imwrite(rima2,fullfile(targetdir_noisy2, filelist2(i).name));
        i
    end
end

%sheng生成噪声图像
if 0
    for i = 3 : fileNum3
        file_name3 = fullfile(maindir3, filelist3(i).name);
        img3 = double(imread(file_name3))/255;
        ima3 = (((img3)));
        %fullfile(targetdir_clean1, filelist1(i).name);
        %imwrite(ima1,fullfile(targetdir_clean1, filelist1(i).name));
        rima3 = imnoise(ima3,'gaussian',0,sigma);
        %rima3 = imnoise(ima3,'speckle',sigma);
        %rima3 = imnoise(ima3,'salt & pepper',sigma);
        imwrite(rima3,fullfile(targetdir_noisy3, filelist3(i).name));
        i
    end
end
