%% Evaluation for LOL / Derain datasets (no masks)
clear; close all; clc

% === Modify these directories ===
shadowdir = 'D:\DAF_IRNet\690\denoise\noise15\output_images\';  % predicted images
freedir   = 'D:\datasets_All\Multi_task\BSDdataset\BSD68\original\';     % ground truth images
% freedir   = 'D:\datasets_All\Multi_task\derain\test\gt\';    % example for derain GT

% === Load image lists ===
SD = dir([shadowdir '/*.png']);   % shadow removal / derain / enhancement results
FD = dir([freedir   '/*.png']);   % ground truth

% === Ensure same number of images ===
numImages = min(length(SD), length(FD));

% === Initialize metrics ===
allrmse = zeros(1, numImages);
ppsnr   = zeros(1, numImages);
sssim   = zeros(1, numImages);

cform = makecform('srgb2lab');

% Accumulators for pixel-pooled RMSE
total_dist_rmse  = 0;
total_pixels_rmse = 0;

%% === Main loop ===
for i = 1:numImages
    sname = fullfile(shadowdir, SD(i).name);
    fname = fullfile(freedir,  FD(i).name);

    s = imread(sname);   % predicted image
    f = imread(fname);   % ground truth

    % Normalize to [0,1]
    f = double(f) / 255;
    s = double(s) / 255;

    % Resize (optional: adjust size to match your training setup)
    s = imresize(s, [256 256]);
    f = imresize(f, [256 256]);

    % ---- Metrics (ALL) ----
    ppsnr(i) = psnr(s, f);
    sssim(i) = ssim(s, f);

    % Convert to Lab and compute RMSE
    f_lab = applycform(f, cform);
    s_lab = applycform(s, cform);
    dist  = (f_lab - s_lab).^2;

    % Per-image RMSE (PI-Lab)
    allrmse(i) = sqrt(sum(dist(:)) / (size(f_lab,1) * size(f_lab,2)));

    % Accumulate for pixel-pooled RMSE (PP-Lab)
    total_dist_rmse  = total_dist_rmse + sum(dist(:));
    total_pixels_rmse = total_pixels_rmse + numel(f_lab);

    disp(i);
end

%% === Print results ===
fprintf('PSNR(all):\n%f\n', mean(ppsnr));
fprintf('SSIM(all):\n%f\n', mean(sssim));
fprintf('RMSE-PI-Lab(all):\n%f\n', mean(allrmse));
fprintf('RMSE-PP-Lab(all):\n%f\n\n', sqrt(total_dist_rmse / total_pixels_rmse));
