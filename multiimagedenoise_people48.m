clear all
%人像去噪48X48
randNum = 10;
sigma1 = 20;
sigma = sigma1/(255*sqrt(5));
%maindir = uigetdir( '选择一个文件夹' );
cleandir = 'D:\\jupyter_notebook\\lena\\clean48\\test';
noisydir = ['D:\\jupyter_notebook\\lena\\noisy48\\gauss\\test\\noisy_test',num2str(sigma1+0.3)];
denoisydir = ['D:\\jupyter_notebook\\lena\\denoise48\\denoise_cum',num2str(sigma1+0.35)];
denoisydir1 = ['D:\\jupyter_notebook\\lena\\denoise48\\denoise',num2str(sigma1+0.35)];

filelist  = dir(cleandir);
filelist_noisy  = dir(noisydir);
filelist_denoisy  = dir(denoisydir);
filelist_denoisy1  = dir(denoisydir1);
%[filelist_denoisy_name,index] = sort_nat(filelist_denoisy.name);
filelist_denoisy_name= [];

fileNum = size(filelist,1)-2; 

NLM_psnr = 0;
Fast_NLM_psnr = 0;
BM3D_psnr = 0;
TV_psnr = 0;
encoder_psnr = 0;
encoder_psnr1 = 0;

NLM_ssim= 0;
Fast_NLM_ssim = 0;
BM3D_ssim = 0;
TV_ssim = 0;
encoder_ssim = 0;
encoder_ssim1 = 0;

randSelect = 3+floor(698*rand(1,randNum)); %从train中随机选择randNum张图片
randSelect
for i = 1 : randNum 
    file_name1 = fullfile(cleandir, filelist(randSelect(i)).name);
    imread(file_name1);
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
    % denoise it
    fima=NLmeansfilter(rima,7,3,sigma);
    fima_f=FAST_NLM_II(rima,5,3,0.15);
    [B_PSNR, fima_BM3D] = BM3D(ima, rima, sigma1, 'np', 0);
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
    TV_psnr = TV_psnr + psnr(fima_TV/255,ima);
    TV_ssim = TV_ssim + ssim(fima_TV/255,ima);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    NLM_psnr = NLM_psnr + psnr(fima,ima);
    Fast_NLM_psnr = Fast_NLM_psnr + psnr(fima_f,ima);
    BM3D_psnr = BM3D_psnr + psnr(fima_BM3D,ima);
    %10*log10(1/mean((ima(:)-double(fima_BM3D(:))).^2));
    NLM_ssim = NLM_ssim + ssim(fima,ima);
    Fast_NLM_ssim = Fast_NLM_ssim + ssim(fima_f,ima);
    BM3D_ssim = BM3D_ssim + ssim(fima_BM3D,ima);
    
    
    filename_encode = [num2str(randSelect(i)-2),'.png'];
    %file_name4 = fullfile(denoisydir1, filelist_denoisy1(randSelect(i)).name);
    s = strcmp({filelist_denoisy.name},filename_encode);
    ind = find(s==1);
    filelist_denoisy(ind).name
    file_name3 = fullfile(denoisydir, filelist_denoisy(ind).name);
    file_name4 = fullfile(denoisydir1, filelist_denoisy1(ind).name);
    eima = rgb2gray(double(imread(file_name3))/255);
    eima1 = rgb2gray(double(imread(file_name4))/255);
    encoder_psnr = encoder_psnr + psnr(eima,ima);
    encoder_psnr1 = encoder_psnr1 + psnr(eima1,ima);
    encoder_ssim = encoder_ssim + ssim(eima,ima);
    encoder_ssim1 = encoder_ssim1 + ssim(eima1,ima);
    i
end

%'result.txt'为文件名；'a'为打开方式：在打开的文件末端添加数据，若文件不存在则创建。
fp=fopen('result1.txt','a');
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
%x = [NLM_psnr NLM_ssim Fast_NLM_psnr Fast_NLM_ssim BM3D_psnr BM3D_ssim TV_psnr TV_ssim...
%    encoder_psnr encoder_ssim encoder_psnr1 encoder_ssim1];
%fp为文件句柄，指定要写入数据的文件。注意：%f后有空格。
%fprintf(fp,'%4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f %4.2f %4.3f\r\n',x);
%fclose(fp);%关闭文件。

disp("NLM  psnr = ");[NLM_psnr,NLM_ssim]
disp("Fast_NLM  psnr = ");[Fast_NLM_psnr,Fast_NLM_ssim];
disp("encoder  psnr = ");[encoder_psnr,encoder_ssim]
disp("encoder1  psnr = ");[encoder_psnr1,encoder_ssim1]
disp("BM3D_psnr = ");[BM3D_psnr,BM3D_ssim]
disp("TV_psnr = ");[TV_psnr,TV_ssim]
figure(2)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_f),title('filtered');
subplot(2,2,4),imagesc(rima-fima_f),title('residuals');

figure(3)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(eima),title('filtered');
subplot(2,2,4),imagesc(ima-eima),title('residuals');

figure(4)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_BM3D),title('filtered');
subplot(2,2,4),imagesc(ima-fima_BM3D),title('residuals');

 figure(5)
 colormap(gray)
 subplot(2,2,1),imagesc(ima),title('original');
 subplot(2,2,2),imagesc(rima),title('noisy');
 subplot(2,2,3),imagesc(eima1),title('filtered');
 subplot(2,2,4),imagesc(ima-eima1),title('residuals');

figure(6)
colormap(gray)
subplot(2,2,1),imagesc(ima),title('original');
subplot(2,2,2),imagesc(rima),title('noisy');
subplot(2,2,3),imagesc(fima_TV),title('filtered');
subplot(2,2,4),imagesc(rima-fima_TV/255),title('residuals');

% num = 10;
% figure(1)
% colormap(gray)
% axis off
% randSelect1 = 3+floor(9997*rand(1,num))
% if sigma1 == 50.1
%     for i = 1 : num
%         file_name1 = fullfile(cleandir, filelist(randSelect1(i)).name);
%         ima = (double(imread(file_name1))/255);
%         if ndims(ima)==3
%             ima = rgb2gray(ima);
%         end
%         file_name2 = fullfile(noisydir, filelist_noisy(randSelect1(i)).name);
%         rima = (double(imread(file_name2))/255);
%         if ndims(rima)==3
%             rima = rgb2gray(rima);
%         end
%         fima=NLmeansfilter(rima,5,2,sigma);
%         fima_f=FAST_NLM_II(rima,5,3,0.5);
%         [B_PSNR, fima_BM3D] = BM3D(ima, rima, 40, 'np', 0);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         lambda = 0.005; % % regularization parameter
%         mu     = 0.05;
%         % % smoother: smaller mu; larger lambda
%         kmax      = 500; % maximum iteration number
%         threshold = 5e-5;
%         fnewname = [file_name2(1:end-4), '.bmp'];
%         imwrite(rima, fnewname, 'bmp');
%         rima_TV = double(imread(fnewname));
%         delete(fnewname);
%         Cparameter = ComputeParameter(rima_TV,1);
%         [fima_TV,Cparameter] = FastATV(rima_TV,rima_TV,Cparameter,lambda,mu,kmax,threshold);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         file_name3 = fullfile(denoisydir, filelist_denoisy(randSelect1(i)).name);
%         file_name4 = fullfile(denoisydir1, filelist_denoisy1(randSelect1(i)).name);
%         eima = rgb2gray(double(imread(file_name3))/255);
%         eima1 = rgb2gray(double(imread(file_name4))/255);
%         subplot(8,num,i),imagesc(fima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+num),imagesc(fima_f);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+2*num),imagesc(fima_BM3D);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+3*num),imagesc(fima_TV);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+4*num),imagesc(eima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+5*num),imagesc(eima1);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+6*num),imagesc(ima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%         subplot(8,num,i+7*num),imagesc(rima);
%         set(gca,'xtick',[],'xticklabel',[])
%         set(gca,'ytick',[],'yticklabel',[])
%     end
% end
%     
