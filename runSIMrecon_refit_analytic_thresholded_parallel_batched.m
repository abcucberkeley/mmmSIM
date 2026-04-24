
% RUNSIMRECON_REFIT_ANALYTIC_THRESHOLDED_PARALLEL
% Canonical end-to-end runner for the current 2-beam 3-phase SIM pipeline.
%
% Uses canonical filenames to avoid path ambiguity:
%   make_otf_2beam_3d_canonical.m
%   refit_sim_params_2beam_3d_analytic_canonical.m
%   sim_recon_2beam_3d_unified_parallel.m
%
% Features:
%   - global analytical parameter refit with thresholded acceptance
%   - load cached OTF unless overwrite_otf=true
%   - unified full-volume or tiled reconstruction
%   - optional local tile refit with score/texture gating
%   - parallel tile refit (CPU) and parallel tile reconstruction
%       * CPU-parallel when useGPU=false
%       * multi-GPU parallel when useGPU=true and >1 GPU available
%       * single-GPU runs sequentially on GPU but still benefits from CPU-parallel tile refit
%   - tile diagnostic figures and FFT diagnostic figures

clear; clc; close all;
tic
%% ---------------- USER SETTINGS ----------------
script_dir = 'X:\abcabc\iSOAR2\Imaging\260409\2026-04-09_isoar_mount_4\test\';
addpath(script_dir);

% Input TIFF files
raw_tiff_file = fullfile(script_dir, 'RAW_560_NA0p5_yaw0_period500nm_DENSE_ch0_CAM1_stack0000_560nm_0000000msec_0007961266msecAbs_-01x_-01y_-01z_0000t.tif');
psf_tiff_file = fullfile(script_dir, 'psf_bbb.tif');

otf_mat_file     = fullfile(script_dir, 'otf_2beam_3d.mat');
recon_tif_file   = fullfile(script_dir, 'recon_2beam_3d.tif');
report_mat_file  = fullfile(script_dir, 'recon_refit_report.mat');
tile_fig_prefix  = fullfile(script_dir, 'tile_diagnostics');
fft_fig_file     = fullfile(script_dir, 'recon_fft_planes.png');

overwrite_otf = false;

nphases      = 3;
ndirs        = 1;
psf_ndirs    = 1;
pxl_dim_data = [0.085, 0.085, 0.100];
pxl_dim_psf  = [0.085, 0.085, 0.100];
wavelength   = 0.525;
na           = 1.35;
nimm         = 1.405;

k0angles      = deg2rad([0]);
linespacing   = 0.494;
phase_step_nm = [];

background   = 150;
wiener       = 0.01;

bead_diameter     = 0;
napodize          = 10;
auto_background   = true;
border_size       = 8;
cleanup_threshold = 0;
truncate_bead_tf_after_first_zero = false;

% cudasirecon-style OTF cleanup / filtering
otf_cleanup_enable = true;
otf_cylindrical_average = true;
otf_fixorigin = [2 9];
otf_leavekz = [0 0 0];
otf_krmax = [];
otf_nocompen = false;
otf_cleanup_interp_method = 'linear';

zoomfact       = 1;
fastSI         = false;
useGPU         = true;
usePerDecomp   = true;
k0_search      = true;
dampenOrder0   = false;
suppress_dc    = false;
gammaApo       = 1;
modamp_thresh  = 0.05;
forcemodamp    = [];
suppress_band_centers = false;
band_suppress_radius_px = 6;
band_suppress_min_weight = 0.05;
band_suppress_power = 6;  % leave empty unless stripes are visible and you intentionally want an override
debug          = false;

% Global refit controls
refit_phase_range_deg      = 5;
refit_phase_step_deg       = 1;
refit_k0_search_window_px  = 6;
auto_accept_refit_if_within_threshold = true;
refit_accept_threshold_pct = 2;

% Tiling controls
use_tiling = true;
tile_size_xyz = [256 256 256];
tile_overlap_xyz = [128 128 128];
tile_refit = true;
tile_refit_accept_threshold_pct = refit_accept_threshold_pct;
tile_auto_accept_refit_if_within_threshold = auto_accept_refit_if_within_threshold;

% Important robustness controls for tiles
tile_local_k0_search = false;   % default false to avoid noise-driven k0 fits in flat tiles
tile_refit_score_min = 0.15;    % reject local refits with weaker overlap score
tile_texture_cv_min  = 0.02;    % reject local refits in nearly uniform tiles

% Parallelization controls
tile_parallel_refit = ~true;
tile_parallel_recon = true;
tile_parallel_workers = [];      % [] = automatic
tile_multigpu = true;            % only used when multiple GPUs are available
tile_single_gpu_batch = true;    % batch same-size/same-params tiles on one GPU
tile_gpu_batch_size = [];        % [] = auto from available GPU memory
tile_batch_min_group = 2;
tile_disable_forcemodamp_on_low_texture = ~true;

% Diagnostic figure controls
save_tile_diagnostic_figures_flag = true;
save_fft_diagnostic_figure_flag   = true;
fft_diag_resample_isotropic       = false;
fft_diag_iso_voxel_um             = [];

% Boundary artifact mitigation
apply_xy_reflect_padding = true;
y_reflect_pad_pixels = 32;
x_reflect_pad_pixels = 32;
apply_z_reflect_padding = true;
z_reflect_pad_planes = 8;

%% ---------------- LOAD TIFF STACKS ----------------
psf_stack = readtiff(psf_tiff_file);
raw_data  = readtiff(raw_tiff_file);
fprintf('Loaded PSF TIFF stack: %s\n', mat2str(size(psf_stack)));
fprintf('Loaded RAW TIFF stack: %s\n', mat2str(size(raw_data)));

if apply_xy_reflect_padding || apply_z_reflect_padding
    [raw_data, padInfo] = reflect_pad_raw_xyz_local(raw_data, fastSI, nphases, ndirs, ...
        y_reflect_pad_pixels, x_reflect_pad_pixels, z_reflect_pad_planes);
    fprintf('Applied reflect padding: y=%d px, x=%d px, z=%d planes on each side\n', ...
        padInfo.padY, padInfo.padX, padInfo.padZ);
else
    padInfo = struct('used', false, 'padY', 0, 'padX', 0, 'padZ', 0);
end

%% ---------------- RESHAPE PSF TIFF ----------------
psf_npages = size(psf_stack, 3);
assert(mod(psf_npages, nphases * psf_ndirs) == 0, 'PSF TIFF page count incompatible with nphases*psf_ndirs');
psf_nz = psf_npages / (nphases * psf_ndirs);
psf_5d = reshape(psf_stack, size(psf_stack,1), size(psf_stack,2), nphases, psf_nz, psf_ndirs);
psf_5d = permute(psf_5d, [1 2 4 3 5]);
if psf_ndirs == 1
    psf_data = psf_5d(:,:,:,:,1);
else
    psf_data = psf_5d;
end
clear psf_stack psf_5d;

%% ---------------- GLOBAL REFIT PARAMETERS FROM RAW DATA ----------------
fit_params = struct();
fit_params.nphases = nphases;
fit_params.ndirs = ndirs;
fit_params.pxl_dim_data = pxl_dim_data;
fit_params.k0angles = k0angles;
fit_params.linespacing = linespacing;
fit_params.background = background;
fit_params.fastSI = fastSI;
fit_params.otf_support_thresh = 0.006;
fit_params.refit_phase_range_deg = refit_phase_range_deg;
fit_params.refit_phase_step_deg = refit_phase_step_deg;
fit_params.refit_k0_search_window_px = refit_k0_search_window_px;

fitReport = refit_sim_params_2beam_3d(raw_data, fit_params);

k0angles_refined      = double(fitReport.k0angles_refined(:)).';
linespacing_refined   = double(fitReport.linespacing_refined(:)).';
phase_vectors_refined = double(fitReport.phase_vectors_rad);

proposed_phase_vectors_rad = build_proposed_phase_vectors(nphases, ndirs, phase_step_nm, linespacing);
proposed_phase_step_rad = effective_phase_step_rad(proposed_phase_vectors_rad);
refined_phase_step_rad  = effective_phase_step_rad(phase_vectors_refined);

angle_dev_deg = abs(rad2deg(wrap_to_pi(k0angles_refined - k0angles)));
angle_dev_pct = 100 * angle_dev_deg / 180;
linespacing_dev_pct = 100 * abs((linespacing_refined - linespacing) ./ max(abs(linespacing), eps));
phase_step_dev_pct  = 100 * abs((refined_phase_step_rad - proposed_phase_step_rad) ./ max(abs(proposed_phase_step_rad), eps));

if auto_accept_refit_if_within_threshold
    accept_angle       = angle_dev_pct <= refit_accept_threshold_pct;
    accept_linespacing = linespacing_dev_pct <= refit_accept_threshold_pct;
    accept_phase       = phase_step_dev_pct <= refit_accept_threshold_pct;
else
    accept_angle       = true(1, ndirs);
    accept_linespacing = true(1, ndirs);
    accept_phase       = true(1, ndirs);
end

k0angles_accepted = k0angles;
k0angles_accepted(accept_angle) = k0angles_refined(accept_angle);
linespacing_accepted = repmat(linespacing, 1, ndirs);
linespacing_accepted(accept_linespacing) = linespacing_refined(accept_linespacing);
phase_vectors_accepted = proposed_phase_vectors_rad;
phase_vectors_accepted(accept_phase, :) = phase_vectors_refined(accept_phase, :);
accepted_phase_step_rad = effective_phase_step_rad(phase_vectors_accepted);
accepted_phase_step_nm  = accepted_phase_step_rad ./ (2*pi) .* linespacing_accepted * 1e3;

fprintf('\nGlobal refit acceptance threshold: %.2f %%\n', refit_accept_threshold_pct);
for d = 1:ndirs
    fprintf('Direction %d\n', d);
    fprintf('  Angle proposed/refined/accepted (deg): %.6f / %.6f / %.6f | dev = %.3f deg (%.3f%%) | accept=%d\n', ...
        rad2deg(k0angles(d)), rad2deg(k0angles_refined(d)), rad2deg(k0angles_accepted(d)), ...
        angle_dev_deg(d), angle_dev_pct(d), accept_angle(d));
    fprintf('  Line spacing proposed/refined/accepted (um): %.6f / %.6f / %.6f | dev = %.3f%% | accept=%d\n', ...
        linespacing, linespacing_refined(d), linespacing_accepted(d), linespacing_dev_pct(d), accept_linespacing(d));
    fprintf('  Phase step proposed/refined/accepted (deg): %.6f / %.6f / %.6f | dev = %.3f%% | accept=%d\n', ...
        rad2deg(proposed_phase_step_rad(d)), rad2deg(refined_phase_step_rad(d)), rad2deg(accepted_phase_step_rad(d)), ...
        phase_step_dev_pct(d), accept_phase(d));
end

%% ---------------- LOAD OR GENERATE OTF ----------------
otf_loaded = false;
otf_meta = struct();
if exist(otf_mat_file, 'file') == 2 && ~overwrite_otf
    try
        S = load(otf_mat_file);
        if isfield(S, 'otf_data')
            otf_data = S.otf_data;
            if isfield(S, 'otf_meta'); otf_meta = S.otf_meta; end
            otf_loaded = true;
            fprintf('Loaded cached OTF from %s\n', otf_mat_file);
        end
    catch ME
        warning('Failed to load cached OTF file. Will regenerate. Reason: %s', ME.message);
    end
end

if ~otf_loaded
    otf_params = struct();
    otf_params.pxl_dim_psf = pxl_dim_psf;
    otf_params.linespacing = linespacing_accepted(1);
    otf_params.k0angles = k0angles_accepted(1:psf_ndirs);
    otf_params.nphases = nphases;
    otf_params.background = [];
    otf_params.auto_background = auto_background;
    otf_params.border_size = border_size;
    otf_params.napodize = napodize;
    otf_params.use_centering = true;
    otf_params.phase_step = [];
    otf_params.phase_vector_rad = phase_vectors_accepted(1:psf_ndirs,:);
    otf_params.bead_diameter = bead_diameter;
    otf_params.min_bead_tf = 0.02;
    otf_params.cleanup_threshold = cleanup_threshold;
    otf_params.tile_to_ndirs = false;
    otf_params.truncate_bead_tf_after_first_zero = truncate_bead_tf_after_first_zero;
    otf_params.wavelength = wavelength;
    otf_params.na = na;
    otf_params.nimm = nimm;
    otf_params.cleanup_otf = otf_cleanup_enable;
    otf_params.cylindrical_average = otf_cylindrical_average;
    otf_params.fixorigin = otf_fixorigin;
    otf_params.leavekz = otf_leavekz;
    otf_params.krmax = otf_krmax;
    otf_params.nocompen = otf_nocompen;
    otf_params.cleanup_interp_method = otf_cleanup_interp_method;
    otf_params.return_cartesian_preview = true;
    otf_params.force_polar_output = false;
    otf_params.output_file = otf_mat_file;

    otf_fun = resolve_otf_fun();
    [otf_data, otf_meta] = otf_fun(psf_data, otf_params);
end

%% ---------------- RECONSTRUCT ----------------
recon_params = struct();
recon_params.nphases       = nphases;
recon_params.ndirs         = ndirs;
recon_params.pxl_dim_data  = pxl_dim_data;
recon_params.pxl_dim_psf   = pxl_dim_psf;
recon_params.wavelength    = wavelength;
recon_params.na            = na;
recon_params.nimm          = nimm;
recon_params.k0angles      = k0angles_accepted;
recon_params.linespacing   = mean(linespacing_accepted);
recon_params.phase_step    = [];
recon_params.phase_vector_rad = phase_vectors_accepted;
recon_params.wiener        = wiener;
recon_params.background    = background;
recon_params.zoomfact      = zoomfact;
recon_params.otfRA         = (otf_num_dirs_local(otf_data) == 1);
recon_params.fastSI        = fastSI;
recon_params.gammaApo      = gammaApo;
recon_params.dampenOrder0  = dampenOrder0;
recon_params.suppress_dc   = suppress_dc;
recon_params.usePerDecomp  = usePerDecomp;
recon_params.k0_search     = k0_search;
recon_params.useGPU        = useGPU;
recon_params.debug         = debug;
recon_params.modamp_thresh = modamp_thresh;
recon_params.forcemodamp   = forcemodamp;

recon_params.use_tiling    = use_tiling;
recon_params.tile_size_xyz = tile_size_xyz;
recon_params.tile_overlap_xyz = tile_overlap_xyz;
recon_params.tile_refit    = tile_refit;
recon_params.tile_refit_accept_threshold_pct = tile_refit_accept_threshold_pct;
recon_params.tile_auto_accept_refit_if_within_threshold = tile_auto_accept_refit_if_within_threshold;
recon_params.tile_local_k0_search = tile_local_k0_search;
recon_params.tile_refit_score_min = tile_refit_score_min;
recon_params.tile_texture_cv_min  = tile_texture_cv_min;
recon_params.tile_parallel_refit = tile_parallel_refit;
recon_params.tile_parallel_recon = tile_parallel_recon;
recon_params.tile_parallel_workers = tile_parallel_workers;
recon_params.tile_multigpu = tile_multigpu;
recon_params.refit_k0_search_window_px = refit_k0_search_window_px;
recon_params.suppress_band_centers = suppress_band_centers;
recon_params.band_suppress_radius_px = band_suppress_radius_px;
recon_params.band_suppress_min_weight = band_suppress_min_weight;
recon_params.band_suppress_power = band_suppress_power;
recon_params.tile_single_gpu_batch = tile_single_gpu_batch;
recon_params.tile_gpu_batch_size = tile_gpu_batch_size;
recon_params.tile_batch_min_group = tile_batch_min_group;
recon_params.tile_disable_forcemodamp_on_low_texture = tile_disable_forcemodamp_on_low_texture;
recon_params.otf_support_thresh = 0.006;
recon_params.sideband_zcut_factor = 1.3;
recon_params.apodizeoutput = 1;
recon_params.amp_in_wiener = true;

[recon, tileReport] = sim_recon_2beam_3d_unified_parallel_batched(raw_data, otf_data, recon_params);

if padInfo.used
    recon = crop_recon_xyz_local(recon, padInfo, zoomfact);
end

writetiff(recon, recon_tif_file);
fprintf('Saved reconstruction to %s\n', recon_tif_file);

%% ---------------- SAVE REPORTS ----------------
acceptanceReport = struct();
acceptanceReport.threshold_pct = refit_accept_threshold_pct;
acceptanceReport.auto_accept = auto_accept_refit_if_within_threshold;
acceptanceReport.k0angles_proposed = k0angles;
acceptanceReport.k0angles_refined = k0angles_refined;
acceptanceReport.k0angles_accepted = k0angles_accepted;
acceptanceReport.angle_dev_deg = angle_dev_deg;
acceptanceReport.angle_dev_pct = angle_dev_pct;
acceptanceReport.accept_angle = accept_angle;
acceptanceReport.linespacing_proposed = repmat(linespacing, 1, ndirs);
acceptanceReport.linespacing_refined = linespacing_refined;
acceptanceReport.linespacing_accepted = linespacing_accepted;
acceptanceReport.linespacing_dev_pct = linespacing_dev_pct;
acceptanceReport.accept_linespacing = accept_linespacing;
acceptanceReport.phase_vectors_proposed = proposed_phase_vectors_rad;
acceptanceReport.phase_vectors_refined = phase_vectors_refined;
acceptanceReport.phase_vectors_accepted = phase_vectors_accepted;
acceptanceReport.phase_step_proposed_rad = proposed_phase_step_rad;
acceptanceReport.phase_step_refined_rad = refined_phase_step_rad;
acceptanceReport.phase_step_accepted_rad = accepted_phase_step_rad;
acceptanceReport.phase_step_accepted_nm = accepted_phase_step_nm;
acceptanceReport.phase_step_dev_pct = phase_step_dev_pct;
acceptanceReport.accept_phase = accept_phase;
acceptanceReport.padInfo = padInfo;
acceptanceReport.zPadInfo = padInfo; % compatibility alias

% save(report_mat_file, 'fitReport', 'acceptanceReport', 'recon_params', 'otf_meta', 'tileReport', '-v7.3');
% fprintf('Saved fit/acceptance report to %s\n', report_mat_file);

% %% ---------------- FIGURES ----------------
% fig = figure('Color','w');
% imagesc(max(recon,[],3));
% axis image off; colormap gray; colorbar;
% title('Reconstruction max projection');
% saveas(fig, fullfile(script_dir, 'recon_max_projection.png'));
% close(fig);
% 
% if use_tiling && isfield(tileReport,'used_tiling') && tileReport.used_tiling && save_tile_diagnostic_figures_flag
%     save_tile_diagnostic_figures(tileReport, tile_fig_prefix);
%     if isfield(tileReport, 'volume_fields')
%         save_volume_diagnostic_figures(tileReport.volume_fields, tile_fig_prefix);
%     end
% end
% 
% if save_fft_diagnostic_figure_flag
%     recon_voxel_xyz = [pxl_dim_data(1)/zoomfact, pxl_dim_data(2)/zoomfact, pxl_dim_data(3)];
%     save_fft_diagnostic_figure(recon, recon_voxel_xyz, fft_fig_file, ...
%         fft_diag_resample_isotropic, fft_diag_iso_voxel_um);
% end
toc
%% ---------------- LOCAL FUNCTIONS ----------------
function phase_vectors = build_proposed_phase_vectors(nphases, ndirs, phase_step_nm, linespacing_um)
if isempty(phase_step_nm)
    base = (0:nphases-1) * (2*pi / nphases);
    phase_vectors = repmat(base, ndirs, 1);
else
    if isscalar(phase_step_nm)
        phase_step_nm = repmat(phase_step_nm, 1, ndirs);
    end
    phase_vectors = zeros(ndirs, nphases);
    for d = 1:ndirs
        delta_phi = (phase_step_nm(d) * 1e-3) / linespacing_um * 2*pi;
        phase_vectors(d,:) = (0:nphases-1) * delta_phi;
    end
end
end

function step_rad = effective_phase_step_rad(phase_vectors_rad)
step_rad = mean(diff(unwrap(phase_vectors_rad, [], 2), 1, 2), 2).';
end

function ang = wrap_to_pi(ang)
ang = mod(ang + pi, 2*pi) - pi;
end

function otf_fun = resolve_otf_fun()
if exist('make_otf_2beam_3d_canonical','file') == 2
    otf_fun = @make_otf_2beam_3d_canonical;
elseif exist('make_otf_2beam_3d','file') == 2
    otf_fun = @make_otf_2beam_3d;
else
    error('Could not find a compatible OTF generator on the MATLAB path.');
end
end

function save_tile_diagnostic_figures(tileReport, outPrefix)
mapsA = tileReport.accepted_maps;
masks = tileReport.accept_masks;
score = tileReport.fit_score;
cvmap = tileReport.texture_cv;
refitUsed = tileReport.tile_refit_used;
[~, ~, nTz, ndirs] = size(mapsA.angle_deg);
midZ = max(1, round(nTz/2));

for d = 1:ndirs
    fig1 = figure('Color','w','Position',[100 100 1400 800]);
    tiledlayout(2,4,'Padding','compact','TileSpacing','compact');

    nexttile; imagesc(mean(mapsA.angle_deg(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted angle (deg), z-mean, dir %d', d));

    nexttile; imagesc(mean(mapsA.linespacing_um(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted line spacing (um), z-mean, dir %d', d));

    nexttile; imagesc(mean(mapsA.phase_step_deg(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted phase step (deg), z-mean, dir %d', d));

    nexttile; imagesc(mean(score(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Fit score, z-mean, dir %d', d));

    nexttile; imagesc(cvmap(:,:,midZ)); axis image; colorbar;
    title(sprintf('Tile texture CV, z-slice %d', midZ));

    nexttile; imagesc(double(refitUsed(:,:,midZ))); axis image; colorbar;
    title(sprintf('Tile refit used, z-slice %d', midZ));

    nexttile; imagesc(double(masks.angle(:,:,midZ,d))); axis image; colorbar;
    title(sprintf('Accept angle mask, z-slice %d, dir %d', midZ, d));

    nexttile; imagesc(double(masks.phase(:,:,midZ,d))); axis image; colorbar;
    title(sprintf('Accept phase mask, z-slice %d, dir %d', midZ, d));

    saveas(fig1, sprintf('%s_tiles_summary_dir%d.png', outPrefix, d));
    close(fig1);
end
end

function save_volume_diagnostic_figures(volumeFields, outPrefix)
mapsA = volumeFields.accepted;
mapsR = volumeFields.refined;
masks = volumeFields.accept_masks;
score = volumeFields.fit_score;
cvmap = volumeFields.texture_cv;
refitUsed = volumeFields.tile_refit_used;
[~, ~, nz, ndirs] = size(mapsA.angle_deg);
midZ = max(1, round(nz/2));

for d = 1:ndirs
    fig1 = figure('Color','w','Position',[100 100 1500 850]);
    tiledlayout(2,5,'Padding','compact','TileSpacing','compact');

    nexttile; imagesc(mean(mapsA.angle_deg(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted angle (deg), z-mean, dir %d', d));
    nexttile; imagesc(mean(mapsA.linespacing_um(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted line spacing (um), z-mean, dir %d', d));
    nexttile; imagesc(mean(mapsA.phase_step_deg(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Accepted phase step (deg), z-mean, dir %d', d));
    nexttile; imagesc(mean(score(:,:,:,d),3,'omitnan')); axis image; colorbar;
    title(sprintf('Fit score, z-mean, dir %d', d));
    nexttile; imagesc(mean(cvmap,3,'omitnan')); axis image; colorbar;
    title('Texture CV, z-mean');

    nexttile; imagesc(mapsA.angle_deg(:,:,midZ,d)); axis image; colorbar;
    title(sprintf('Accepted angle z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsA.linespacing_um(:,:,midZ,d)); axis image; colorbar;
    title(sprintf('Accepted line spacing z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsA.phase_step_deg(:,:,midZ,d)); axis image; colorbar;
    title(sprintf('Accepted phase step z=%d dir %d', midZ, d));
    nexttile; imagesc(score(:,:,midZ,d)); axis image; colorbar;
    title(sprintf('Fit score z=%d dir %d', midZ, d));
    nexttile; imagesc(double(refitUsed(:,:,midZ))); axis image; colorbar;
    title(sprintf('Tile refit used z=%d', midZ));

    saveas(fig1, sprintf('%s_volume_summary_dir%d.png', outPrefix, d));
    close(fig1);

    fig2 = figure('Color','w','Position',[100 100 1200 350]);
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imagesc(double(masks.angle(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept angle z=%d dir %d', midZ, d));
    nexttile; imagesc(double(masks.linespacing(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept spacing z=%d dir %d', midZ, d));
    nexttile; imagesc(double(masks.phase(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept phase z=%d dir %d', midZ, d));
    saveas(fig2, sprintf('%s_volume_accept_masks_dir%d.png', outPrefix, d));
    close(fig2);

    fig3 = figure('Color','w','Position',[100 100 1200 700]);
    tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
    nexttile; imagesc(mean(mapsR.angle_deg(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Refined angle z-mean dir %d', d));
    nexttile; imagesc(mean(mapsR.linespacing_um(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Refined spacing z-mean dir %d', d));
    nexttile; imagesc(mean(mapsR.phase_step_deg(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Refined phase z-mean dir %d', d));
    nexttile; imagesc(mapsR.angle_deg(:,:,midZ,d)); axis image; colorbar; title(sprintf('Refined angle z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsR.linespacing_um(:,:,midZ,d)); axis image; colorbar; title(sprintf('Refined spacing z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsR.phase_step_deg(:,:,midZ,d)); axis image; colorbar; title(sprintf('Refined phase z=%d dir %d', midZ, d));
    saveas(fig3, sprintf('%s_volume_refined_dir%d.png', outPrefix, d));
    close(fig3);
end
end

function save_fft_diagnostic_figure(recon, voxel_xyz, outFile, doIso, isoVoxel)
vol = single(recon);
vox = double(voxel_xyz(:).');
if doIso
    if isempty(isoVoxel)
        isoVoxel = min(vox);
    end
    if exist('imresize3', 'file') == 2
        newSize = max(1, round(size(vol) .* (vox / isoVoxel)));
        vol = imresize3(vol, newSize, 'linear');
        vox = [isoVoxel isoVoxel isoVoxel];
    else
        warning('imresize3 not available. Skipping isotropic resampling for FFT diagnostics.');
    end
end

F = fftshift(fftn(vol));
Fmag = abs(F);
[ny, nx, nz] = size(Fmag);
cy = floor(ny/2) + 1;
cx = floor(nx/2) + 1;
cz = floor(nz/2) + 1;
dc = Fmag(cy, cx, cz);
if dc <= eps('single')
    dc = max(Fmag(:));
end
if dc <= eps('single')
    dc = single(1);
end
Fnorm = Fmag ./ dc;
Fshow = log10(1 + Fnorm);

kx = normalized_k_axis(nx, vox(2));
ky = normalized_k_axis(ny, vox(1));
kz = normalized_k_axis(nz, vox(3));

fig = figure('Color','w','Position',[100 100 1400 450]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(kx, ky, squeeze(Fshow(:,:,cz))); axis image; colorbar;
xlabel('k_x / k_{Nyq,x}'); ylabel('k_y / k_{Nyq,y}');
title(sprintf('log_{10}(1+|F|/DC), k_x-k_y @ k_z=0'));

nexttile;
imagesc(kx, kz, squeeze(Fshow(cy,:,:)).'); axis image; colorbar;
xlabel('k_x / k_{Nyq,x}'); ylabel('k_z / k_{Nyq,z}');
title(sprintf('log_{10}(1+|F|/DC), k_x-k_z @ k_y=0'));

nexttile;
imagesc(ky, kz, squeeze(Fshow(:,cx,:)).'); axis image; colorbar;
xlabel('k_y / k_{Nyq,y}'); ylabel('k_z / k_{Nyq,z}');
title(sprintf('log_{10}(1+|F|/DC), k_y-k_z @ k_x=0'));

saveas(fig, outFile);
close(fig);
end

function knull = normalized_k_axis(N, d)
idx = -ceil((N-1)/2):floor((N-1)/2);
k = idx / (N * d);
knyq = 0.5 / d;
knull = k / knyq;
end


function data5d = raw3d_to_5d_local(raw_data, fastSI, nphases, ndirs)
[ny, nx, nimgs] = size(raw_data);
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1) == 0, 'raw_data size incompatible with nphases*ndirs');
nz = round(nz);
if fastSI
    data5d = reshape(raw_data, ny, nx, nphases, ndirs, nz);
    data5d = permute(data5d, [1 2 5 3 4]);
else
    data5d = reshape(raw_data, ny, nx, nphases, nz, ndirs);
    data5d = permute(data5d, [1 2 4 3 5]);
end
end

function raw3d = raw5d_to_3d_local(data5d, fastSI)
[ny, nx, ~, ~, ~] = size(data5d);
if fastSI
    tmp = permute(data5d, [1 2 4 5 3]);
else
    tmp = permute(data5d, [1 2 4 3 5]);
end
raw3d = reshape(tmp, ny, nx, []);
end


function [raw_padded, info] = reflect_pad_raw_xyz_local(raw_data, fastSI, nphases, ndirs, padY, padX, padZ)
[ny, nx, nimgs] = size(raw_data);
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1) == 0, 'raw_data size incompatible with nphases*ndirs');
nz = round(nz);

padY = max(0, min(round(padY), ny-1));
padX = max(0, min(round(padX), nx-1));
padZ = max(0, min(round(padZ), nz-1));

if padY == 0 && padX == 0 && padZ == 0
    raw_padded = raw_data;
    info = struct('used', false, 'padY', 0, 'padX', 0, 'padZ', 0, ...
                  'ny_original', ny, 'nx_original', nx, 'nz_original', nz, ...
                  'ny_padded', ny, 'nx_padded', nx, 'nz_padded', nz);
    return;
end

data5d = raw3d_to_5d_local(raw_data, fastSI, nphases, ndirs);

if padY > 0
    preY  = data5d(padY:-1:1,:,:,:,:);
    postY = data5d(end:-1:end-padY+1,:,:,:,:);
    data5d = cat(1, preY, data5d, postY);
end
if padX > 0
    preX  = data5d(:,padX:-1:1,:,:,:);
    postX = data5d(:,end:-1:end-padX+1,:,:,:);
    data5d = cat(2, preX, data5d, postX);
end
if padZ > 0
    preZ  = data5d(:,:,padZ:-1:1,:,:);
    postZ = data5d(:,:,end:-1:end-padZ+1,:,:);
    data5d = cat(3, preZ, data5d, postZ);
end

raw_padded = raw5d_to_3d_local(data5d, fastSI);

info = struct('used', true, 'padY', padY, 'padX', padX, 'padZ', padZ, ...
              'ny_original', ny, 'nx_original', nx, 'nz_original', nz, ...
              'ny_padded', size(data5d,1), 'nx_padded', size(data5d,2), 'nz_padded', size(data5d,3));
end

function recon = crop_recon_xyz_local(recon_padded, padInfo, zoomfact)
yPad = padInfo.padY * zoomfact;
xPad = padInfo.padX * zoomfact;
zPad = padInfo.padZ;

y1 = 1 + yPad;
y2 = size(recon_padded,1) - yPad;
x1 = 1 + xPad;
x2 = size(recon_padded,2) - xPad;
z1 = 1 + zPad;
z2 = size(recon_padded,3) - zPad;

recon = recon_padded(y1:y2, x1:x2, z1:z2);
end



function n = otf_num_dirs_local(otf_data)
if isstruct(otf_data) && isfield(otf_data,'ndirs')
    n = double(otf_data.ndirs);
else
    sz = size(otf_data);
    if numel(sz) < 5
        n = 1;
    else
        n = sz(5);
    end
end
end
