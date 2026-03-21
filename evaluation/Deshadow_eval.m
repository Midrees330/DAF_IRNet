%% compute PSNR, SSIM (RGB) and RMSE (LAB) 
clear;close all;clc
% Modify the following directories
shadowdir = 'D:\DAF_IRNet\690\shadow\output_images\';
maskdir = 'D:\datasets_All\ISTD+\dataset\test\test_B\';
freedir = 'D:\datasets_All\ISTD+\dataset\test\test_C\';

MD = dir([maskdir '/*.png']);
SD = dir([shadowdir '/*.png']);
FD = dir([freedir '/*.png']);

total_dists_rmse = 0;
total_pixels_rmse = 0;
total_distn_rmse = 0;
total_pixeln_rmse = 0;

allrmse=zeros(1,size(SD,1)); 
srmse=zeros(1,size(SD,1)); 
nrmse=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    
    f = double(f)/255;
    s = double(s)/255;
    
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);

    nmask=~m;       %mask of non-shadow region
    smask=~nmask;   %mask of shadow regions
    
    %% PSNR and SSIM in RGB space
    ppsnr(i)=psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

    %% Convert to LAB for RMSE computation
    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% RMSE in LAB space
    dist_rmse=(f - s).^2;
    sdist_rmse=dist_rmse.*repmat(smask,[1 1 3]);
    sumsdist_rmse=sum(sdist_rmse(:));
    ndist_rmse=dist_rmse.*repmat(nmask,[1 1 3]);
    sumndist_rmse=sum(ndist_rmse(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% RMSE per image
    allrmse(i)=sqrt(sum(dist_rmse(:))/(size(f,1)*size(f,2)));
    srmse(i)=sqrt(sumsdist_rmse/sumsmask);
    nrmse(i)=sqrt(sumndist_rmse/sumnmask);
    
    total_dists_rmse = total_dists_rmse + sumsdist_rmse;
    total_pixels_rmse = total_pixels_rmse + sumsmask * 3;
    
    total_distn_rmse = total_distn_rmse + sumndist_rmse;
    total_pixeln_rmse = total_pixeln_rmse + sumnmask * 3;  

    disp(i);
end
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
fprintf('RMSE-PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allrmse),mean(nrmse),mean(srmse));
fprintf('RMSE-PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allrmse),sqrt(total_distn_rmse/total_pixeln_rmse),sqrt(total_dists_rmse/total_pixels_rmse));