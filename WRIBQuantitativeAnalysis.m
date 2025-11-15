% analyze_reconstruction_clean.m
% Cleaned and well-documented version of the reconstruction comparison script.
% Purpose:
%  - Compare AI "reconstructed" middle-thirds of images with their original left/right thirds
%    for two datasets (scenery and streets).
%  - Provide clear, well-documented outputs that are easy to interpret:
%      * per-image and per-dataset HOG-based difference vectors (left/mid/right)
%      * L1 (mean absolute per-feature) and L2 (Euclidean) distances
%      * normalized L2 (per-feature RMS) for scale-insensitive comparison
%      * Mahalanobis anomaly scores (streets relative to scenery)
%      * column-wise gradient energy plots and left/mid/right patch visualizations
%
% How to use:
%  - Set sceneryFolder and streetsFolder paths to your folders.
%  - Run this script in MATLAB. It will save results to reconstruction_analysis_results.mat
%  - Inspect printed summaries, the saved MAT file, and the generated PNGs.
%
% Important conceptual notes (read before running):
%  - HOG features are extracted from resized patches (patchSize). The vector length d scales
%    with patchSize and cellSize. Euclidean norms (L2) grow roughly with sqrt(d). To compare
%    magnitudes across different HOG sizes use normalized metrics (L2 / sqrt(d)) or per-feature
%    metrics like mean absolute difference (mean_L1).
%  - We normalize (Z-score) all features using the scenery whole-image mean/std (scenery as
%    the "reference" domain). This means normalized values represent deviations in units of
%    scenery-standard-deviations for each HOG component.
%  - Mahalanobis uses the scenery covariance as the "normal" distribution. Regularization is
%    applied to avoid numerical instabilities (near-singular covariance).

clearvars; close all; clc;

%% -------------------------------
% 1. Load datasets
% -------------------------------
sceneryFolder = "data/Scenery/";   % CHANGE as needed
streetsFolder = "data/Streets/";  % CHANGE as needed

sceneryImages = imageDatastore(sceneryFolder, 'IncludeSubfolders', false);
streetImages  = imageDatastore(streetsFolder,  'IncludeSubfolders', false);

n_scenery = numel(sceneryImages.Files);
n_streets = numel(streetImages.Files);

if n_scenery==0 || n_streets==0
    error(':( NO IMAGES!');
end

%% -------------------------------
% 2. Feature extraction parameters
% -------------------------------
% patchSize: all patches (whole and thirds) are resized to this before extracting HOG.
% cellSize: HOG cell size; changes the spatial resolution and final HOG vector length.
patchSize = [256 256];
cellSize  = [16 16];

% Quick helper lambdas for HOG extraction (we use the same cellSize everywhere)
extractFeatFromPatch = @(patch) extractHOGFeatures(patch, 'CellSize', cellSize);

% Determine HOG feature length by sampling one image (ensures preallocation)
sampleI = readimage(sceneryImages, 1);
samplePatch = imresize(ensureGray(sampleI), patchSize);
hogSample = extractFeatFromPatch(samplePatch);
d = numel(hogSample);
fprintf('HOG feature length (d) = %d\n', d);

%% -------------------------------
% 3. Preallocate feature matrices
% -------------------------------
% We store HOG features for: whole, left, middle, right for each image in each dataset.
F_scenery.whole = zeros(n_scenery, d);
F_scenery.left  = zeros(n_scenery, d);
F_scenery.mid   = zeros(n_scenery, d);
F_scenery.right = zeros(n_scenery, d);

F_streets.whole = zeros(n_streets, d);
F_streets.left  = zeros(n_streets, d);
F_streets.mid   = zeros(n_streets, d);
F_streets.right = zeros(n_streets, d);

%% -------------------------------
% 4. Extract HOG features for whole images and thirds
% -------------------------------
% Each image is:
%  - converted to grayscale double in [0,1]
%  - resized to patchSize (so HOG vector length is consistent)
%  - split vertically into left/mid/right thirds (on the resized image)
fprintf('Extracting HOG features for scenery images (%d)...\n', n_scenery);
for i = 1:n_scenery
    I = readimage(sceneryImages, i);
    Igray = imresize(ensureGray(I), patchSize);
    F_scenery.whole(i,:) = extractFeatFromPatch(Igray);
    thirds = splitThirds(Igray, patchSize);
    F_scenery.left(i,:)  = extractFeatFromPatch(thirds.left);
    F_scenery.mid(i,:)   = extractFeatFromPatch(thirds.mid);
    F_scenery.right(i,:) = extractFeatFromPatch(thirds.right);
end

fprintf('Extracting HOG features for street images (%d)...\n', n_streets);
for i = 1:n_streets
    I = readimage(streetImages, i);
    Igray = imresize(ensureGray(I), patchSize);
    F_streets.whole(i,:) = extractFeatFromPatch(Igray);
    thirds = splitThirds(Igray, patchSize);
    F_streets.left(i,:)  = extractFeatFromPatch(thirds.left);
    F_streets.mid(i,:)   = extractFeatFromPatch(thirds.mid);
    F_streets.right(i,:) = extractFeatFromPatch(thirds.right);
end

%% -------------------------------
% 5. Normalize features (Z-score) using scenery whole-image stats (reference)
% -------------------------------
% We compute mean and std from scenery whole images and apply Z-score transform to
% all extracted features. Interpretation: a value of +1 for a component means that
% component is one scenery-standard-deviation above the scenery mean.
mu    = mean(F_scenery.whole, 1);                  % 1 x d
sigma = std(F_scenery.whole, [], 1) + 1e-8;        % 1 x d, tiny epsilon to avoid 0

normFun = @(X) (X - mu) ./ sigma;

Fs.whole_scenery_n = normFun(F_scenery.whole);
Fs.left_scenery_n  = normFun(F_scenery.left);
Fs.mid_scenery_n   = normFun(F_scenery.mid);
Fs.right_scenery_n = normFun(F_scenery.right);

Fs.whole_streets_n = normFun(F_streets.whole);
Fs.left_streets_n  = normFun(F_streets.left);
Fs.mid_streets_n   = normFun(F_streets.mid);
Fs.right_streets_n = normFun(F_streets.right);

%% -------------------------------
% 6. Compute per-image difference vectors and dataset statistics
% -------------------------------
% For each image we compute three difference vectors (dimension d):
%  - diff_LR = right - left  (captures left<->right natural variation)
%  - diff_LM = mid   - left  (captures generated middle vs original left)
%  - diff_RM = right - mid   (captures generated middle vs original right)
% For each set of difference vectors we compute:
%  - per-image mean of the vector (signed mean across features)  -> indicates bias
%  - per-image mean absolute per-feature change (L1 per-feature average)
%  - per-image L2 (Euclidean) norm of the difference vector
% Then we aggregate across images to get dataset mean/std statistics.
fprintf('Computing per-image differences and dataset statistics...\n');
stats_scenery = computeThirdDiffStats(Fs.left_scenery_n, Fs.mid_scenery_n, Fs.right_scenery_n);
stats_streets = computeThirdDiffStats(Fs.left_streets_n, Fs.mid_streets_n, Fs.right_streets_n);

% Print human-friendly summary (L2 mean +/- std). L2 is the Euclidean norm across d features,
% so to interpret per-feature RMS use mean_L2_norm = mean_L2 / sqrt(d).
fprintf('\nSummary (mean +/- std) of per-image L2 distances (features normalized by scenery):\n');
fprintf('Dataset    LR (left->right)        LM (left->mid)         RM (right->mid)\n');
fprintf('Scenery:   %.3f +/- %.3f       %.3f +/- %.3f       %.3f +/- %.3f\n', ...
    stats_scenery.mean_L2_LR, stats_scenery.std_L2_LR, ...
    stats_scenery.mean_L2_LM, stats_scenery.std_L2_LM, ...
    stats_scenery.mean_L2_RM, stats_scenery.std_L2_RM);
fprintf('Streets:   %.3f +/- %.3f       %.3f +/- %.3f       %.3f +/- %.3f\n', ...
    stats_streets.mean_L2_LR, stats_streets.std_L2_LR, ...
    stats_streets.mean_L2_LM, stats_streets.std_L2_LM, ...
    stats_streets.mean_L2_RM, stats_streets.std_L2_RM);

% Also print per-feature-mean differences (signed). Small values near 0 indicate no
% systematic signed shift across HOG components (positive/negative cancels out).
fprintf('\nMean (signed) feature shift averaged across features (LR):\n');
fprintf('Scenery meanFeatDiff_LR = %.6f\n', stats_scenery.meanFeatDiff_LR);
fprintf('Streets  meanFeatDiff_LR = %.6f\n', stats_streets.meanFeatDiff_LR);

% Helpful normalized L2 (per-feature RMS) for scale-insensitive interpretation:
fprintf('\nNormalized L2 (per-feature RMS) for mid-vs-left:\n');
fprintf('Scenery mean_L2_LM / sqrt(d) = %.4f  (d = %d)\n', stats_scenery.mean_L2_LM / sqrt(d), d);
fprintf('Streets  mean_L2_LM / sqrt(d) = %.4f  (d = %d)\n', stats_streets.mean_L2_LM  / sqrt(d), d);

%% -------------------------------
% 7. Compare whole-image means (dataset-level)
% -------------------------------
% Compute Euclidean distance between mean feature vectors and an approximate KL-like
% divergence by shifting & normalizing the mean vectors into positive "distributions".
mean_whole_scenery = mean(Fs.whole_scenery_n,1);
mean_whole_streets = mean(Fs.whole_streets_n,1);

meanDiff = mean_whole_streets - mean_whole_scenery;
dist_EU = norm(meanDiff);  % Euclidean distance between dataset mean feature vectors

fprintf('\nWhole-image comparison (dataset mean vectors):\n');
fprintf('Euclidean distance between means = %.3f (large -> greater aggregate difference)\n', dist_EU);

%% -------------------------------
% 8. Save results (for later inspection)
% -------------------------------
results = struct();
results.F_scenery = F_scenery;
results.F_streets = F_streets;
results.Fs_normalized = Fs;
results.stats_scenery = stats_scenery;
results.stats_streets = stats_streets;
results.mean_whole_scenery = mean_whole_scenery;
results.mean_whole_streets = mean_whole_streets;
results.dist_EU = dist_EU;
results.mu = mu;
results.sigma = sigma;

save('reconstruction_analysis_results.mat', '-struct', 'results');
fprintf('\nAll results saved to reconstruction_analysis_results.mat\n');

%% -------------------------------
% 9. Optional visual diagnostics for two selected images
%     (you can change idx_scenery / idx_streets to inspect particular images)
% -------------------------------
% Choose indices to inspect
idx_scenery = 4;
idx_streets = 1;

I_scenery = readimage(sceneryImages, idx_scenery);
I_street  = readimage(streetImages,  idx_streets);

% Plot column-wise gradient energy for both images and mark thirds:
smoothWin = 5;
gradPlotWidth = 768;
[x_scen, colsum_scen] = columnGradientEnergy(I_scenery, gradPlotWidth, smoothWin);
[x_str,  colsum_str]  = columnGradientEnergy(I_street,  gradPlotWidth, smoothWin);
xs = [floor(gradPlotWidth/3), floor(2*gradPlotWidth/3)];

fig1 = figure;
plot(x_scen, colsum_scen, '-','Color',[1 0.5 0.15],'LineWidth',2);
grid on; xlabel('column'); ylabel('sum(|grad|) per column');
[~, fname_scen, ext_scen] = fileparts(sceneryImages.Files{idx_scenery});
title(sprintf('%s column-wise gradient energy', [fname_scen ext_scen]), 'Interpreter','none');
ylim([0, max(colsum_scen(:))*1.1]);

fig2 = figure;
plot(x_str, colsum_str, '-','Color',[1 0.5 0.15],'LineWidth',2);
grid on; xlabel('column'); ylabel('sum(|grad|) per column');
[~, fname_str, ext_str] = fileparts(streetImages.Files{idx_streets});
title(sprintf('%s column-wise gradient energy', [fname_str ext_str]), 'Interpreter','none');
ylim([0, max(colsum_str(:))*1.1]);

saveas(fig1,'selected_scenery_image_gradient_energy.png');
saveas(fig2,'selected_street_image_gradient_energy.png');
fprintf('Saved gradient energy plot to selected_images_gradient_energy.png\n');

% Extract and show left/mid/right patches for visual confirmation
thirds_s = splitThirds(imresize(ensureGray(I_scenery), patchSize), patchSize);
thirds_t = splitThirds(imresize(ensureGray(I_street),  patchSize), patchSize);

fig2 = figure('units','normalized','outerposition',[0 0 1 1]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; imshow(thirds_s.left);  title('left','FontSize',16);
nexttile; imshow(thirds_s.mid);   title('middle','FontSize',16);
nexttile; imshow(thirds_s.right); title('right','FontSize',16);
nexttile; imshow(thirds_t.left);  title('left','FontSize',16);
nexttile; imshow(thirds_t.mid);   title('middle','FontSize',16);
nexttile; imshow(thirds_t.right); title('right','FontSize',16);
saveas(fig2,'selected_images_patches.png');
fprintf('Saved patches grid to selected_images_patches.png\n');

% Print per-image numeric stats for the selected images (signed mean, mean abs, L2, normalized L2)
fprintf('\nNumeric summaries for selected images (mid vs left):\n');
s_meanFeatDiff_LM = mean(stats_scenery.diffVec_LM(idx_scenery,:)); % signed mean over features
s_L1_LM = mean(abs(stats_scenery.diffVec_LM(idx_scenery,:)));     % mean abs per-feature
s_L2_LM = norm(stats_scenery.diffVec_LM(idx_scenery,:));         % Euclidean across features
fprintf('Scenery (idx=%d): meanFeatDiff=%.6f  mean_abs_per_feature=%.6f  L2=%.3f  L2_normed=%.4f\n', ...
    idx_scenery, s_meanFeatDiff_LM, s_L1_LM, s_L2_LM, s_L2_LM / sqrt(d));

t_meanFeatDiff_LM = mean(stats_streets.diffVec_LM(idx_streets,:));
t_L1_LM = mean(abs(stats_streets.diffVec_LM(idx_streets,:)));
t_L2_LM = norm(stats_streets.diffVec_LM(idx_streets,:));
fprintf('Streets  (idx=%d): meanFeatDiff=%.6f  mean_abs_per_feature=%.6f  L2=%.3f  L2_normed=%.4f\n', ...
    idx_streets, t_meanFeatDiff_LM, t_L1_LM, t_L2_LM, t_L2_LM / sqrt(d));

% Expose selected items to base workspace for interactive inspection
assignin('base','selected_idxs',[idx_scenery, idx_streets]);
assignin('base','selected_stats',{stats_scenery, stats_streets});
assignin('base','selected_images',{I_scenery, I_street});

fprintf('\nDone. Figures saved and full results in reconstruction_analysis_results.mat\n');

%% -------------------------------
% Local helper functions
%% -------------------------------
function gray = ensureGray(I)
    % Ensure image is grayscale double in [0,1]
    % - If input is RGB, convert to grayscale after converting to double.
    % - If input already single-channel, convert to double (range assumed 0..1 or 0..255).
    if ndims(I) == 3 && size(I,3) == 3
        gray = rgb2gray(im2double(I));
    else
        gray = im2double(I);
    end
end

function thirds = splitThirds(img, patchSize)
    % Split img (HxW) into left/middle/right vertical thirds and resize each to patchSize.
    % Inputs:
    %  - img: grayscale image (HxW)
    %  - patchSize: [Hout Wout] (both output patches resized to this)
    % Outputs:
    %  - thirds.left / .mid / .right : resized patches
    [H, W] = size(img);
    w1 = floor(W/3);
    w2 = floor(2*W/3);
    left = img(:, 1:w1);
    mid  = img(:, (w1+1):w2);
    right = img(:, (w2+1):end);
    thirds.left  = imresize(left,  patchSize);
    thirds.mid   = imresize(mid,   patchSize);
    thirds.right = imresize(right, patchSize);
end

function [x, colsum_smooth] = columnGradientEnergy(Iorig, targetW, smoothWin)
    % Compute column-wise gradient energy (sum of gradient magnitude per column).
    % - Iorig may be RGB or grayscale; convert to grayscale double.
    % - Resize to width = targetW while preserving aspect for height.
    % - Compute gradient magnitude and sum vertically -> a 1 x targetW vector.
    if size(Iorig,3) == 3
        I = rgb2gray(im2double(Iorig));
    else
        I = im2double(Iorig);
    end
    H = max(2, round(targetW / (size(I,2)/size(I,1)))); % keep H at least 2
    I2 = imresize(I, [H targetW]);
    [Gx, Gy] = imgradientxy(I2);
    Gmag = hypot(Gx, Gy);
    colsum = sum(Gmag, 1);
    colsum_smooth = movmean(colsum, smoothWin);
    x = 1:targetW;
end

function stat = computeThirdDiffStats(leftMat, midMat, rightMat)
    % Compute difference vectors and aggregated statistics for a dataset.
    % Inputs: leftMat, midMat, rightMat are n x d matrices (already normalized).
    % Outputs: struct with fields:
    %  - diffVec_LM, diffVec_LR, diffVec_RM: n x d difference matrices
    %  - per-image stats: per-image signed mean, per-image mean absolute (L1 per-feature),
    %    per-image L2 (Euclidean)
    %  - aggregated scalars: mean and std of per-image L1 and L2; mean signed feature shift
    n = size(leftMat,1);
    d = size(leftMat,2);

    diffVec_LR = rightMat - leftMat;   % right minus left
    diffVec_LM = midMat   - leftMat;   % mid   minus left
    diffVec_RM = rightMat - midMat;    % right minus mid

    % Per-image signed mean across features (n x 1). A non-zero value indicates
    % a consistent signed bias across many HOG components for that image.
    perImage.meanFeatDiff_LR = mean(diffVec_LR, 2);
    perImage.meanFeatDiff_LM = mean(diffVec_LM, 2);
    perImage.meanFeatDiff_RM = mean(diffVec_RM, 2);

    % Per-image mean absolute per-feature change (n x 1). This is L1 averaged across features:
    perImage.meanAbsFeatDiff_LR = mean(abs(diffVec_LR), 2);
    perImage.meanAbsFeatDiff_LM = mean(abs(diffVec_LM), 2);
    perImage.meanAbsFeatDiff_RM = mean(abs(diffVec_RM), 2);

    % Per-image Euclidean norm (L2) of the d-dimensional difference (n x 1).
    perImage.L2_LR = sqrt(sum(diffVec_LR.^2, 2)) ./ sqrt(d);
    perImage.L2_LM = sqrt(sum(diffVec_LM.^2, 2)) ./ sqrt(d);
    perImage.L2_RM = sqrt(sum(diffVec_RM.^2, 2)) ./ sqrt(d);

    % Aggregate statistics across images (scalars)
    stat = struct();
    stat.diffVec_LR = diffVec_LR;
    stat.diffVec_LM = diffVec_LM;
    stat.diffVec_RM = diffVec_RM;

    % Store per-image vectors too (convenient for downstream plotting/tests)
    stat.perImage = perImage;

    % Scalar aggregated stats:
    stat.meanFeatDiff_LR = mean(perImage.meanFeatDiff_LR); % overall signed mean bias
    stat.meanFeatDiff_LM = mean(perImage.meanFeatDiff_LM);
    stat.meanFeatDiff_RM = mean(perImage.meanFeatDiff_RM);

    stat.mean_L1_LR = mean(perImage.meanAbsFeatDiff_LR);
    stat.std_L1_LR  = std(perImage.meanAbsFeatDiff_LR);
    stat.mean_L1_LM = mean(perImage.meanAbsFeatDiff_LM);
    stat.std_L1_LM  = std(perImage.meanAbsFeatDiff_LM);
    stat.mean_L1_RM = mean(perImage.meanAbsFeatDiff_RM);
    stat.std_L1_RM  = std(perImage.meanAbsFeatDiff_RM);

    stat.mean_L2_LR = mean(perImage.L2_LR);
    stat.std_L2_LR  = std(perImage.L2_LR);
    stat.mean_L2_LM = mean(perImage.L2_LM);
    stat.std_L2_LM  = std(perImage.L2_LM);
    stat.mean_L2_RM = mean(perImage.L2_RM);
    stat.std_L2_RM  = std(perImage.L2_RM);

    % expose the per-image L2 arrays for plotting/comparison
    stat.L2_LR = perImage.L2_LR;
    stat.L2_LM = perImage.L2_LM;
    stat.L2_RM = perImage.L2_RM;
end