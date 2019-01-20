clear all
%clc
%clf
%在minist数据集上进行各方法的比较
randNum = 1;
sigma1 = 50;
sigma = (sigma1/255)^2;
%maindir = uigetdir( '选择一个文件夹' );
cleandir = ['D:\\jupyter_notebook\\minist\\denoise\\clean_test\\'];
%cleandir = 'D:\\jupyter_notebook\\minist\\noisy\\test\\clean_test';
noisydir = ['D:\\jupyter_notebook\\minist\\denoise\\noisy_test',num2str(sigma1+0.1)];
%noisydir = ['D:\\jupyter_notebook\\minist\\noisy\\test\\noisy_test',num2str(sigma)]
denoisydir = ['D:\\jupyter_notebook\\minist\\denoise\\denoise_normal',num2str(sigma1+0.15)];
denoisydir1 = ['D:\\jupyter_notebook\\minist\\denoise\\denoise_cum',num2str(sigma1+0.15)];
filelist  = dir(cleandir);
filelist_noisy  = dir(noisydir);
filelist_denoisy  = dir(denoisydir);
filelist_denoisy1  = dir(denoisydir1);
fileNum = size(filelist,1)-2; 

NLM_psnr = 0;
Fast_NLM_psnr = 0;
BM3D_psnr = 0;
TV_psnr = 0;
encoder_psnr = 0;
encoder_psnr1 = 0;
% 可在ssim和RFSIM指标间切换
NLM_ssim= 0;
Fast_NLM_ssim = 0;
BM3D_ssim = 0;
TV_ssim = 0;
encoder_ssim = 0;
encoder_ssim1 = 0;

randSelect = 3+floor(997*rand(1,randNum)); %从noisy_test中随机选择randNum张图片
t1 = zeros(1,randNum);
t2 = zeros(1,randNum);
t3 = zeros(1,randNum);
t4 = zeros(1,randNum);
gauss = zeros(6,3);
imagenoise = zeros(784*randNum,6);
for i = 1 : randNum 
    file_name1 = fullfile(cleandir, filelist(randSelect(i)).name);
    ima = (double(imread(file_name1))/255);
    if ndims(ima)==3
        ima = rgb2gray(ima);
    end
    file_name2 = fullfile(noisydir, filelist_noisy(randSelect(i)).name);
    rima = (double(imread(file_name2))/255);
    if ndims(rima)==3
        rima = rgb2gray(rima);
    end
    %rima = imnoise(ima,'salt & pepper',0.01);
    %rima = imnoise(ima,'gaussian',0,sigma);
    %imwrite(rima,fullfile(py_dir, filelist(randSelect(i) - 2).name));
    % denoise 过程
    tic
    fima=NLmeansfilter(rima,5,2,sigma);
    temp = (ima-fima);
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    imagenoise((i-1)*784+1:i*784,1) = imagenoise((i-1)*784+1:i*784,1)+temp(1:end)';
    %gauss(1,:) = gauss(1,:) + sg;
    t1(i) = toc;
    tic
    %fima_f=FAST_NLM_II(rima,5,3,0.15);
    fima_f = medfilt2(rima, [2,2]);%中值滤波
    temp = (ima-fima_f);
    imagenoise((i-1)*784+1:i*784,2) = imagenoise((i-1)*784+1:i*784,2)+temp(1:end)';
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    %gauss(2,:) = gauss(2,:) + sg;
    t2(i) = toc;
    tic
    [B_PSNR, fima_BM3D] = BM3D(ima, rima, 30, 'np', 0);
    temp = (ima-fima_BM3D);
    imagenoise((i-1)*784+1:i*784,3) = imagenoise((i-1)*784+1:i*784,3)+temp(1:end)';
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    %gauss(3,:) = gauss(3,:) + sg;         
    t3(i) = toc;
    tic
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lambda = 0.005; % % regularization parameter
    mu     = 0.05;
    % % smoother: smaller mu; larger lambda
    kmax      = 500; % maximum iteration number
    threshold = 5e-5;
    fnewname = [file_name2(1:end-4), '.bmp'];
    imwrite(rima, fnewname, 'bmp');
    rima_TV = double(imread(fnewname));
    delete(fnewname);
    Cparameter = ComputeParameter(rima_TV,1);
    [fima_TV,Cparameter] = FastATV(rima_TV,rima_TV,Cparameter,lambda,mu,kmax,threshold);
    t4(i) = toc;
    temp = (ima-fima_TV/255);
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    %gauss(4,:) = gauss(4,:) + sg;
    imagenoise((i-1)*784+1:i*784,4) = imagenoise((i-1)*784+1:i*784,4)+temp(1:end)';    
    TV_psnr = TV_psnr + psnr(fima_TV/255,ima);
    TV_ssim = TV_ssim + ssim(fima_TV/255,ima);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    NLM_psnr = NLM_psnr + psnr(fima,ima);
    Fast_NLM_psnr = Fast_NLM_psnr + psnr(fima_f,ima);
    BM3D_psnr = BM3D_psnr + psnr(fima_BM3D,ima);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %NLM_ssim = NLM_ssim + ssim(fima,ima);
    NLM_ssim = NLM_ssim + RFSIM(ima,fima); 
    %Fast_NLM_ssim = Fast_NLM_ssim + ssim(fima_f,ima);
    Fast_NLM_ssim = Fast_NLM_ssim + RFSIM(ima,fima_f);
    %BM3D_ssim = BM3D_ssim + ssim(fima_BM3D,ima);
    BM3D_ssim = BM3D_ssim + RFSIM(ima,fima_BM3D);
    
    file_name3 = fullfile(denoisydir, filelist_denoisy(randSelect(i)).name);
    file_name4 = fullfile(denoisydir1, filelist_denoisy1(randSelect(i)).name);
    eima = rgb2gray(double(imread(file_name3))/255);
    temp = (ima-eima);
    imagenoise((i-1)*784+1:i*784,5) = imagenoise((i-1)*784+1:i*784,5)+temp(1:end)';
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    %if sg(3)>0.5
    %    gauss(5,:) = gauss(5,:) + 1;
    %end
    %gauss(5,:) = gauss(5,:) + sg;
    
    eima1 = rgb2gray(double(imread(file_name4))/255);
    temp = (ima-eima1);
    imagenoise((i-1)*784+1:i*784,6) = imagenoise((i-1)*784+1:i*784,6)+temp(1:end)';
    %[sg, sl] = glstat(temp(1:end)',0.51,128);
    %if sg(3)>0.5
    %    gauss(6,:) = gauss(6,:) + 1;
    %end
    %gauss(6,:) = gauss(6,:) + sg;
    encoder_psnr = encoder_psnr + psnr(eima,ima);
    encoder_psnr1 = encoder_psnr1 + psnr(eima1,ima);
    %encoder_ssim = encoder_ssim + ssim(eima,ima);
    encoder_ssim = encoder_ssim + RFSIM(ima,eima);
    %encoder_ssim1 = encoder_ssim1 + ssim(eima1,ima);
    encoder_ssim1 = encoder_ssim1 + RFSIM(ima,eima1);
end
[sg, sl] = glstat(imagenoise(:,1),0.51,128);
gauss(1,:) = gauss(1,:) + sg;

[sg, sl] = glstat(imagenoise(:,2),0.51,128);
gauss(2,:) = gauss(2,:) + sg;

[sg, sl] = glstat(imagenoise(:,3),0.51,128);
gauss(3,:) = gauss(3,:) + sg;         

[sg, sl] = glstat(imagenoise(:,4),0.51,128);
gauss(4,:) = gauss(4,:) + sg;
[sg, sl] = glstat(imagenoise(:,5),0.51,128);
gauss(5,3) = gauss(5,3) + sg(1,3);
gauss(5,1) = gauss(5,1) + var(imagenoise(:,5));
[sg, sl] = glstat(imagenoise(:,6),0.51,128);
gauss(6,3) = gauss(6,3) + sg(1,3);
gauss(6,1) = gauss(6,1) + var(imagenoise(:,6));

%'result.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
fp=fopen('result2.txt','a');
NLM_psnr = NLM_psnr / randNum;
Fast_NLM_psnr = Fast_NLM_psnr / randNum;
BM3D_psnr = BM3D_psnr / randNum;
TV_psnr = TV_psnr / randNum;
encoder_psnr = encoder_psnr / randNum;
encoder_psnr1 = encoder_psnr1 / randNum;

NLM_ssim = NLM_ssim / randNum;
Fast_NLM_ssim = Fast_NLM_ssim / randNum;
BM3D_ssim = BM3D_ssim / randNum;
TV_ssim = TV_ssim / randNum;
encoder_ssim = encoder_ssim / randNum;
encoder_ssim1 = encoder_ssim1 / randNum;
x = [NLM_psnr NLM_ssim Fast_NLM_psnr Fast_NLM_ssim BM3D_psnr BM3D_ssim TV_psnr TV_ssim...
    encoder_psnr encoder_ssim encoder_psnr1 encoder_ssim1];
gauss = gauss';
%fp为文件句柄，指定要写入数据的文件。注意：%f后有空格。
fprintf(fp,'%4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f\r\n',x);
fprintf(fp,'%4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f\r\n',gauss(1,:),gauss(3,:));
fclose(fp);%关闭文件。

disp("NLM  psnr = ");
disp("Fast_NLM  psnr = ");
disp("encoder  psnr = ");
disp("encoder1  psnr = ");
disp("BM3D_psnr = ");
disp("TV_psnr = ");
[mean(t1),mean(t2),mean(t3),mean(t4)]
figure(1)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima),title('filtered');
subplot(2,2,4),imagesc(ima-fima),title('residuals');
figure(2)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_f),title('filtered');
subplot(2,2,4),imagesc(ima-fima_f),title('residuals');


figure(5)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(eima),title('filtered');
subplot(2,2,4),imagesc(ima-eima),title('residuals');

figure(3)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_BM3D),title('filtered');
subplot(2,2,4),imagesc(ima-fima_BM3D),title('residuals');

figure(6)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(eima1),title('filtered');
subplot(2,2,4),imagesc(ima-eima1),title('residuals');


figure(4)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_TV),title('filtered');
subplot(2,2,4),imagesc(ima-fima_TV/255),title('residuals');


num = 6;%并排画出处理后的图像样本
compare = 4;
figure(4)
colormap(gray)
axis off
randSelect1 = 3+floor(997*rand(1,num))
if sigma1 == 50
    for i = 1 : num
        %干净图像
        file_name1 = fullfile(cleandir, filelist(randSelect1(i)).name);
        ima = (double(imread(file_name1))/255);
        if ndims(ima)==3
            ima = rgb2gray(ima);
        end
        %噪声图像
        file_name2 = fullfile(noisydir, filelist_noisy(randSelect1(i)).name);
        rima = (double(imread(file_name2))/255);
        if ndims(rima)==3
            rima = rgb2gray(rima);
        end
        %去噪图像
        fima=NLmeansfilter(rima,5,2,sigma);
        fima_f=FAST_NLM_II(rima,5,3,0.5);
        [B_PSNR, fima_BM3D] = BM3D(ima, rima, 40, 'np', 0);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        lambda = 0.005; % % regularization parameter
        mu     = 0.05;
        % % smoother: smaller mu; larger lambda
        kmax      = 500; % maximum iteration number
        threshold = 5e-5;
        fnewname = [file_name2(1:end-4), '.bmp'];
        imwrite(rima, fnewname, 'bmp');
        rima_TV = double(imread(fnewname));
        delete(fnewname);
        Cparameter = ComputeParameter(rima_TV,1);
        [fima_TV,Cparameter] = FastATV(rima_TV,rima_TV,Cparameter,lambda,mu,kmax,threshold);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        file_name3 = fullfile(denoisydir, filelist_denoisy(randSelect1(i)).name);
        file_name4 = fullfile(denoisydir1, filelist_denoisy1(randSelect1(i)).name);
        eima = rgb2gray(double(imread(file_name3))/255);
        eima1 = rgb2gray(double(imread(file_name4))/255);
%         subplot(8,num,i),imagesc(fima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(5,num,i+num),imagesc(fima_f);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+2*num),imagesc(fima_BM3D);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+3*num),imagesc(fima_TV);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+1*num),imagesc(eima);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+2*num),imagesc(eima1);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+3*num),imagesc(ima);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+4*num),imagesc(rima);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
    end
    figure(7)
    colormap(gray)
    for i = 1 : num
        %干净图像
        file_name1 = fullfile(cleandir, filelist(randSelect1(i)).name);
        ima = (double(imread(file_name1))/255);
        if ndims(ima)==3
            ima = rgb2gray(ima);
        end
        %噪声图像
        file_name2 = fullfile(noisydir, filelist_noisy(randSelect1(i)).name);
        rima = (double(imread(file_name2))/255);
        if ndims(rima)==3
            rima = rgb2gray(rima);
        end
        %去噪图像
        fima=NLmeansfilter(rima,5,2,sigma);
        fima_f=FAST_NLM_II(rima,5,3,0.5);
        [B_PSNR, fima_BM3D] = BM3D(ima, rima, 40, 'np', 0);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        lambda = 0.005; % % regularization parameter
        mu     = 0.05;
        % % smoother: smaller mu; larger lambda
        kmax      = 500; % maximum iteration number
        threshold = 5e-5;
        fnewname = [file_name2(1:end-4), '.bmp'];
        imwrite(rima, fnewname, 'bmp');
        rima_TV = double(imread(fnewname));
        delete(fnewname);
        Cparameter = ComputeParameter(rima_TV,1);
        [fima_TV,Cparameter] = FastATV(rima_TV,rima_TV,Cparameter,lambda,mu,kmax,threshold);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        file_name3 = fullfile(denoisydir, filelist_denoisy(randSelect1(i)).name);
        file_name4 = fullfile(denoisydir1, filelist_denoisy1(randSelect1(i)).name);
        eima = rgb2gray(double(imread(file_name3))/255);
        eima1 = rgb2gray(double(imread(file_name4))/255);
        subplot(5,num,i),imagesc(fima);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+num),imagesc(fima_f);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+2*num),imagesc(fima_BM3D);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
        subplot(5,num,i+3*num),imagesc(fima_TV);
        set(gca,'xtick',[],'xticklabel',[])
        set(gca,'ytick',[],'yticklabel',[])
%         subplot(5,num,i+1*num),imagesc(eima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(5,num,i+2*num),imagesc(eima1);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(5,num,i+3*num),imagesc(ima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(5,num,i+4*num),imagesc(rima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
    end
end
    
