% RUNSIMRECON_FOLDER_MULTIGPU_CHANNELS
% Batch folder runner for vertically striped multi-channel TIFFs.
%
% New channel-crop features:
%   - one raw TIFF can contain 1-4 channels stored as vertical stripes
%   - user can enable/disable channel cropping
%   - each channel can define:
%       * x_range  [x1 x2]
%       * y_range  [y1 y2]
%       * z_range  [z1 z2]   in Z-planes, not TIFF page indices
%       * psf_file  path to the PSF TIFF for that channel
%   - each channel is saved under:
%       <output_subfolder>/ch0/
%       <output_subfolder>/ch1/
%       ...
%     using the SAME raw filename
%
% Existing behavior is preserved when use_channel_crops = false.

clear; clc; close all;
tic
%% ---------------- USER SETTINGS ----------------
script_dir = '/clusterfs/vast/abcabc/iSOAR2/Imaging/260417/2026-04-17_isoar_mount_1b/fish/fish1/roi2/';
script_dir = '/clusterfs/vast/abcabc/iSOAR2/Imaging/260410/2026-04-10_isoar_mount_2/fish1/roi1/';
% script_dir = 'X:\abcabc\iSOAR2\Imaging\260410\2026-04-10_isoar_mount_3\fish1\roi1\'
psf_dir = '/clusterfs/nvme2/Data/iSOAR2_nvme2/OTF/';
% psf_dir = 'X:\abcabc\iSOAR2\Imaging\260409\2026-04-09_isoar_mount_4\test\'
addpath(script_dir);

% Input locations
raw_input_folder = script_dir;
psf_input = fullfile(psf_dir, '488_20px_2beam_3phase_PSF.tif');   % used only when use_channel_crops = false

% Global camera offset image (2D) applied to the full raw frame BEFORE channel cropping.
% Set to '' to disable. The image must match the full raw frame size [ny nx].
camera_offset_file = '/clusterfs/vast/abcabc/iSOAR2/Background/260423/2304x2304_14ms_AVG_488_560.tif';

% Output location
output_subfolder_name = 'GPUsiReconM';
output_root = fullfile(raw_input_folder, output_subfolder_name);

% Channel crop mode
use_channel_crops = true;

% Per-channel definitions.
% x_range / y_range / z_range:
%   [] means full range
%   z_range is in Z-planes, not TIFF page indices
%
% Example for vertically striped channels:
channel_configs = struct([]);

channel_configs(1).enabled  = true;
channel_configs(1).name     = 'ch0_488';
channel_configs(1).x_range  = [4 2304];   % e.g. [1 512]
channel_configs(1).y_range  = [777 1531];
channel_configs(1).z_range  = [1 41-4];
% channel_configs(1).psf_file = fullfile(psf_dir, 'RAW_Scan_Iter_0000_ch0_CAM1_stack0000_488nm_0000000msec_0001963574msecAbs_-01x_-01y_-01z_0000t.tif');
channel_configs(1).psf_file = fullfile(psf_dir, '488_20px_2beam_3phase_PSF.tif');
channel_configs(1).background = [10];      % [] => use global background
channel_configs(1).wiener = [0.05];          % [] => use global wiener
channel_configs(1).camera_offset_file = '';  % legacy per-channel field; ignored when global camera_offset_file is used


channel_configs(2).enabled  = true;
channel_configs(2).name     = 'ch1_560';
channel_configs(2).x_range  = [1 2301];%channel_configs(1).x_range;
channel_configs(2).y_range  = [3 757];
channel_configs(2).z_range  = channel_configs(1).z_range;
% channel_configs(2).psf_file = fullfile(psf_dir, 'RAW_560nm_singlebead_SIM-PSF_ch0_CAM1_stack0000_560nm_0000000msec_0015140412msecAbs_-01x_-01y_-01z_0000t.tif');
channel_configs(2).psf_file = fullfile(psf_dir, '488_20px_2beam_3phase_PSF.tif');
channel_configs(2).background = [10];
channel_configs(2).wiener = [0.1];
channel_configs(2).camera_offset_file = '';


channel_configs(3).enabled  = false;
channel_configs(3).name     = 'ch2';
channel_configs(3).x_range  = channel_configs(1).x_range;
channel_configs(3).y_range  = [];
channel_configs(3).z_range  = channel_configs(1).z_range;
channel_configs(3).psf_file = fullfile(psf_dir, 'psf_ch2.tif');
channel_configs(3).background = [];
channel_configs(3).wiener = [];
channel_configs(3).camera_offset_file = '';

channel_configs(4).enabled  = false;
channel_configs(4).name     = 'ch3';
channel_configs(4).x_range  = channel_configs(1).x_range;
channel_configs(4).y_range  = [];
channel_configs(4).z_range  = channel_configs(1).z_range;
channel_configs(4).psf_file = fullfile(psf_dir, 'psf_ch3.tif');
channel_configs(4).background = [];
channel_configs(4).wiener = [];
channel_configs(4).camera_offset_file = '';

% Function names to call
recon_function_candidates = { ...
    'sim_recon_2beam_3d_unified_parallel_batched', ...
    };

otf_function_candidates = { ...
    'make_otf_2beam_3d_canonical', ...
    };

refit_function_candidates = { ...
    'refit_sim_params_2beam_3d', ...
    };

% Batch policy
overwrite_otf   = false;
overwrite_recon = true;
parallelize_across_files = true;
max_file_workers = [];

% GPU/CPU worker scheduling
precompute_shared_otfs = true;    % build/load shared OTFs once up front
use_process_pool_per_gpu = true;  % process-based pool, one worker per GPU
gpu_worker_num_threads = [];      % [] = auto = floor(Processes.NumWorkers / nGPUWorkers)

% -------- Current user parameters (unchanged) --------
nphases      = 3;
ndirs        = 1;
psf_ndirs    = 1;
pxl_dim_data = [0.085, 0.085, 0.250];
pxl_dim_psf  = [0.085, 0.085, 0.1];
wavelength   = 0.525;
na           = 1.2;
nimm         = 1.405;

k0angles      = deg2rad([0]);
linespacing   = 0.504;
phase_step_nm = [];% [0, linespacing/3*1000, linespacing/3*1000*(2)];

background   = 100;
background_psf = [];
wiener       = 0.1;

bead_diameter     = 0;
napodize          = 10;
auto_background   = true;
border_size       = 8;
cleanup_threshold = 0;
truncate_bead_tf_after_first_zero = false;

% cudasirecon-style OTF cleanup / filtering
otf_cleanup_enable = true;
otf_cylindrical_average = true;
otf_fixorigin = [3 20];
otf_leavekz = [1 -1 0];
otf_krmax = [];
otf_nocompen = false;
otf_cleanup_interp_method = 'linear';
return_cartesian_preview = true;
force_polar_output = false;


zoomfact       = 1;
fastSI         = true;
useGPU         = true;
usePerDecomp   = true;
k0_search      = true;
dampenOrder0   = false;
suppress_dc    = false;
gammaApo       = 1;
modamp_thresh  = 0.05;
forcemodamp    = [1];
suppress_band_centers = false;
band_suppress_radius_px = 6;
band_suppress_min_weight = 0.005;
band_suppress_power = 6;
debug          = false;
save16bit      = true;

% Global refit controls
refit_phase_range_deg      = 20;
refit_phase_step_deg       = 1;
refit_k0_search_window_px  = 6;
auto_accept_refit_if_within_threshold = true;
refit_accept_threshold_pct = 5;

% Tiling controls
use_tiling = ~true;
tile_size_xyz = [64 64 64];
tile_overlap_xyz = [16 16 16];
tile_refit = false;
tile_refit_accept_threshold_pct = refit_accept_threshold_pct;
tile_auto_accept_refit_if_within_threshold = auto_accept_refit_if_within_threshold;

% Tile robustness / performance controls
tile_local_k0_search = false;
tile_refit_score_min = 0.15;
tile_texture_cv_min  = 0.02;
tile_parallel_refit = true;
tile_parallel_recon = true;
tile_parallel_workers = [];
tile_multigpu = ~true;
tile_single_gpu_batch = true;
tile_gpu_batch_size = [];
tile_batch_min_group = 2;
tile_disable_forcemodamp_on_low_texture = ~true;

% Diagnostic figure controls
save_tile_diagnostic_figures_flag = ~true;
save_fft_diagnostic_figure_flag   = ~true;
fft_diag_resample_isotropic       = false;
fft_diag_iso_voxel_um             = [];

% Spatial background/haze subtraction
use_spatial_background = ~true;
background_mode = 'opening2d_meanphase';   % {'none','opening2d_meanphase','gaussian3d_meanphase'}
background_open_radius_xy_px = 50;
background_smooth_z_sigma_px = 1.0;
background_gaussian_sigma_xy_px = 32;
background_gaussian_sigma_z_px  = 2.0;

% Boundary artifact mitigation
apply_xy_reflect_padding = true;
y_reflect_pad_pixels = 64;
x_reflect_pad_pixels = 192;
apply_z_reflect_padding = apply_xy_reflect_padding;
z_reflect_pad_planes = 12;

%% ---------------- PREP ----------------
assert(exist(raw_input_folder, 'dir') == 7, 'raw_input_folder does not exist.');

if exist(output_root, 'dir') ~= 7
    mkdir(output_root);
end

if use_channel_crops
    enabled = find([channel_configs.enabled]);
    assert(~isempty(enabled), 'use_channel_crops=true but no enabled channel_configs were found.');
    for k = enabled
        chdir = fullfile(output_root, channel_configs(k).name);
        if exist(chdir, 'dir') ~= 7
            mkdir(chdir);
        end
        assert(exist(resolve_path_local(channel_configs(k).psf_file, script_dir), 'file') == 2, ...
            'PSF file for %s does not exist: %s', channel_configs(k).name, channel_configs(k).psf_file);
    end
else
    assert(exist(psf_input, 'file') == 2 || exist(psf_input, 'dir') == 7, ...
        'psf_input must be a TIFF file or folder when use_channel_crops=false');
end

recon_function_name = resolve_first_available(recon_function_candidates);
otf_function_name   = resolve_first_available(otf_function_candidates);
refit_function_name = resolve_first_available(refit_function_candidates);

fprintf('Using recon function: %s\n', recon_function_name);
fprintf('Using OTF function:   %s\n', otf_function_name);
fprintf('Using refit function: %s\n', refit_function_name);

raw_files = list_tiff_files(raw_input_folder);
assert(~isempty(raw_files), 'No TIFF files found in raw_input_folder.');

% Exclude PSF filenames from the raw list if they live in the same folder
exclude_names = {};
if ~use_channel_crops
    if exist(psf_input, 'file') == 2
        [~, a, b] = fileparts(psf_input);
        exclude_names{end+1} = [a b];
    end
else
    enabled = find([channel_configs.enabled]);
    for k = enabled
        pf = resolve_path_local(channel_configs(k).psf_file, script_dir);
        if exist(pf, 'file') == 2
            [~, a, b] = fileparts(pf);
            exclude_names{end+1} = [a b];
        end
    end
end
exclude_names = unique(exclude_names);

if ~isempty(exclude_names)
    keep = true(numel(raw_files),1);
    for i = 1:numel(raw_files)
        keep(i) = ~any(strcmpi(raw_files(i).name, exclude_names));
    end
    raw_files = raw_files(keep);
end
assert(~isempty(raw_files), 'After excluding PSF files, no raw TIFF files remain.');

cfg = struct();
cfg.script_dir = script_dir;
cfg.nphases = nphases;
cfg.ndirs = ndirs;
cfg.psf_ndirs = psf_ndirs;
cfg.pxl_dim_data = pxl_dim_data;
cfg.pxl_dim_psf = pxl_dim_psf;
cfg.wavelength = wavelength;
cfg.na = na;
cfg.nimm = nimm;
cfg.k0angles = k0angles;
cfg.linespacing = linespacing;
cfg.phase_step_nm = phase_step_nm;
cfg.background = background;
cfg.wiener = wiener;
cfg.bead_diameter = bead_diameter;
cfg.napodize = napodize;
cfg.auto_background = auto_background;
cfg.border_size = border_size;
cfg.cleanup_threshold = cleanup_threshold;
cfg.truncate_bead_tf_after_first_zero = truncate_bead_tf_after_first_zero;
cfg.zoomfact = zoomfact;
cfg.fastSI = fastSI;
cfg.useGPU = useGPU;
cfg.usePerDecomp = usePerDecomp;
cfg.k0_search = k0_search;
cfg.dampenOrder0 = dampenOrder0;
cfg.suppress_dc = suppress_dc;
cfg.gammaApo = gammaApo;
cfg.modamp_thresh = modamp_thresh;
cfg.forcemodamp = forcemodamp;
cfg.suppress_band_centers = suppress_band_centers;
cfg.band_suppress_radius_px = band_suppress_radius_px;
cfg.band_suppress_min_weight = band_suppress_min_weight;
cfg.band_suppress_power = band_suppress_power;
cfg.debug = debug;
cfg.refit_phase_range_deg = refit_phase_range_deg;
cfg.refit_phase_step_deg = refit_phase_step_deg;
cfg.refit_k0_search_window_px = refit_k0_search_window_px;
cfg.auto_accept_refit_if_within_threshold = auto_accept_refit_if_within_threshold;
cfg.refit_accept_threshold_pct = refit_accept_threshold_pct;
cfg.use_tiling = use_tiling;
cfg.tile_size_xyz = tile_size_xyz;
cfg.tile_overlap_xyz = tile_overlap_xyz;
cfg.tile_refit = tile_refit;
cfg.tile_refit_accept_threshold_pct = tile_refit_accept_threshold_pct;
cfg.tile_auto_accept_refit_if_within_threshold = tile_auto_accept_refit_if_within_threshold;
cfg.tile_local_k0_search = tile_local_k0_search;
cfg.tile_refit_score_min = tile_refit_score_min;
cfg.tile_texture_cv_min = tile_texture_cv_min;
cfg.tile_parallel_refit = tile_parallel_refit;
cfg.tile_parallel_recon = tile_parallel_recon;
cfg.tile_parallel_workers = tile_parallel_workers;
cfg.tile_multigpu = tile_multigpu;
cfg.tile_single_gpu_batch = tile_single_gpu_batch;
cfg.tile_gpu_batch_size = tile_gpu_batch_size;
cfg.tile_batch_min_group = tile_batch_min_group;
cfg.tile_disable_forcemodamp_on_low_texture = tile_disable_forcemodamp_on_low_texture;
cfg.save_tile_diagnostic_figures_flag = save_tile_diagnostic_figures_flag;
cfg.save_fft_diagnostic_figure_flag = save_fft_diagnostic_figure_flag;
cfg.fft_diag_resample_isotropic = fft_diag_resample_isotropic;
cfg.fft_diag_iso_voxel_um = fft_diag_iso_voxel_um;
cfg.use_spatial_background = use_spatial_background;
cfg.background_mode = background_mode;
cfg.background_open_radius_xy_px = background_open_radius_xy_px;
cfg.background_smooth_z_sigma_px = background_smooth_z_sigma_px;
cfg.background_gaussian_sigma_xy_px = background_gaussian_sigma_xy_px;
cfg.background_gaussian_sigma_z_px = background_gaussian_sigma_z_px;
cfg.apply_xy_reflect_padding = apply_xy_reflect_padding;
cfg.y_reflect_pad_pixels = y_reflect_pad_pixels;
cfg.x_reflect_pad_pixels = x_reflect_pad_pixels;
cfg.apply_z_reflect_padding = apply_z_reflect_padding;
cfg.z_reflect_pad_planes = z_reflect_pad_planes;
cfg.overwrite_otf = overwrite_otf;
cfg.overwrite_recon = overwrite_recon;
cfg.parallelize_across_files = parallelize_across_files;
cfg.output_root = output_root;
cfg.psf_input = psf_input;
cfg.recon_function_name = recon_function_name;
cfg.otf_function_name = otf_function_name;
cfg.refit_function_name = refit_function_name;
cfg.use_channel_crops = use_channel_crops;
cfg.channel_configs = channel_configs;
cfg.save16bit = save16bit;
cfg.otf_cleanup_enable = otf_cleanup_enable;
cfg.otf_cylindrical_average = otf_cylindrical_average;
cfg.otf_fixorigin = otf_fixorigin;
cfg.otf_leavekz = otf_leavekz;
cfg.otf_krmax = otf_krmax;
cfg.otf_nocompen = otf_nocompen;
cfg.otf_cleanup_interp_method = otf_cleanup_interp_method;
cfg.background_psf = background_psf;
cfg.return_cartesian_preview = return_cartesian_preview;
cfg.force_polar_output = force_polar_output;
cfg.camera_offset_file = camera_offset_file;
cfg.camera_offset_image = [];
cfg.precompute_shared_otfs = precompute_shared_otfs;
cfg.use_process_pool_per_gpu = use_process_pool_per_gpu;
cfg.gpu_worker_num_threads = gpu_worker_num_threads;

jobs = repmat(struct(), numel(raw_files), 1);
for i = 1:numel(raw_files)
    jobs(i).raw_file = fullfile(raw_files(i).folder, raw_files(i).name);
    jobs(i).raw_name = raw_files(i).name;
end

fprintf('Found %d raw TIFF file(s).\n', numel(jobs));
fprintf('Reconstructions will be saved under: %s\n', output_root);

if cfg.precompute_shared_otfs
    cfg = prepare_shared_channel_otfs_once(cfg);
    % Shared OTFs are now prepared; channel/file workers should load them, not rebuild them.
    cfg.overwrite_otf = false;
end
cfg = preload_global_camera_offset_once(cfg);

%% ---------------- RUN BATCH ----------------
statuses = cell(numel(jobs),1);

nGPUs = available_gpu_count();
canPar = can_use_parfor();
doFileParallelGPU = parallelize_across_files && useGPU && nGPUs > 1 && canPar;

if doFileParallelGPU
    nWorkers = min_nonempty(max_file_workers, min(numel(jobs), nGPUs));

    if cfg.use_process_pool_per_gpu
        pool = ensure_gpu_process_pool(nWorkers, cfg.gpu_worker_num_threads);
    else
        pool = ensure_process_pool(nWorkers);
    end

    parfor i = 1:numel(jobs)
        assign_worker_gpu(nGPUs);
        statuses{i} = run_one_job_multichannel(jobs(i), cfg, true);
    end
else
    if useGPU && nGPUs >= 1
        assign_main_gpu(1);
    end
    for i = 1:numel(jobs)
        statuses{i} = run_one_job_multichannel(jobs(i), cfg, false);
    end
end

%% ---------------- SUMMARY ----------------
nOK = 0;
for i = 1:numel(statuses)
    st = statuses{i};
    if ~isempty(st) && isfield(st,'ok') && st.ok
        nOK = nOK + 1;
    end
end
fprintf('\nCompleted %d / %d file(s) successfully.\n', nOK, numel(jobs));
for i = 1:numel(statuses)
    st = statuses{i};
    if isempty(st), continue; end
    if st.ok
        fprintf('  OK   %s\n', st.raw_name);
    else
        fprintf('  FAIL %s\n', st.raw_name);
        fprintf('       %s\n', st.message);
        fprintf('       log: %s\n', st.log_file);
    end
end

%% ---------------- LOCAL FUNCTIONS ----------------
function st = run_one_job_multichannel(job, cfg, fileLevelParallelGPU)
st = struct('ok', false, 'raw_name', job.raw_name, 'message', '', 'log_file', '');

raw_log_file = fullfile(cfg.output_root, [strip_extension(job.raw_name), '_folderrun_log.txt']);
st.log_file = raw_log_file;
fid = fopen(raw_log_file, 'w');
cleanupObj = onCleanup(@() safe_fclose(fid)); %#ok<NASGU>

try
    fprintf(fid, 'Processing raw file: %s\n', job.raw_file);
    raw_data_full = readtiff(job.raw_file);

    if ~isempty(cfg.camera_offset_image)
        raw_data_full = apply_global_camera_offset_local(raw_data_full, cfg.camera_offset_image);
        fprintf(fid, 'Applied global camera offset before cropping.\n');
    end

    if cfg.use_channel_crops
        enabled = find([cfg.channel_configs.enabled]);
        data5d_full = raw3d_to_5d_local(raw_data_full, cfg.fastSI, cfg.nphases, cfg.ndirs);
        clear raw_data_full;
        for ii = 1:numel(enabled)
            ch = cfg.channel_configs(enabled(ii));
            [raw_data, cropInfo] = crop_raw_channel_from_5d_local(data5d_full, ch, cfg.fastSI);

            chdir = fullfile(cfg.output_root, ch.name);
            if exist(chdir, 'dir') ~= 7
                mkdir(chdir);
            end

            channelJob = build_channel_job(job.raw_name, ch, chdir, cfg);
            process_one_channel(channelJob, raw_data, cropInfo, cfg, fileLevelParallelGPU, fid);
        end
        clear data5d_full;
    else
        channelJob = build_default_job(job.raw_name, cfg);
        process_one_channel(channelJob, raw_data_full, struct('used_crop', false), cfg, fileLevelParallelGPU, fid);
    end

    st.ok = true;
    st.message = 'OK';
catch ME
    st.ok = false;
    st.message = ME.message;
    fprintf(fid, 'ERROR: %s\n', ME.message);
    for kk = 1:numel(ME.stack)
        fprintf(fid, '  at %s (line %d)\n', ME.stack(kk).name, ME.stack(kk).line);
    end
end
end

function process_one_channel(job, raw_data, cropInfo, cfg, fileLevelParallelGPU, fid)
fprintf(fid, '\n=== Channel/job: %s ===\n', job.channel_name);
fprintf(fid, 'Output recon: %s\n', job.output_recon);
if cropInfo.used_crop
    fprintf(fid, 'Crop y=[%d %d], x=[%d %d], z=[%d %d]\n', cropInfo.y1, cropInfo.y2, cropInfo.x1, cropInfo.x2, cropInfo.z1, cropInfo.z2);
end

if exist(job.output_recon, 'file') == 2 && ~cfg.overwrite_recon
    fprintf(fid, 'Skipping existing reconstruction.\n');
    return;
end

[raw_data, corrInfo] = apply_channel_corrections_local(raw_data, job.channel_background);
background_for_fit_and_recon = 0;
fprintf(fid, 'Applied channel corrections: scalar background=%g\n', ...
    corrInfo.scalar_background);

if cfg.use_spatial_background
    [raw_data, bgInfo] = subtract_spatial_background_local(raw_data, cfg.fastSI, cfg.nphases, cfg.ndirs, ...
        cfg.background_mode, cfg.background_open_radius_xy_px, cfg.background_smooth_z_sigma_px, ...
        cfg.background_gaussian_sigma_xy_px, cfg.background_gaussian_sigma_z_px);
    fprintf(fid, 'Applied spatial background subtraction: mode=%s\n', bgInfo.mode);
else
    bgInfo = struct('used', false, 'mode', 'none');
end

if cfg.apply_xy_reflect_padding || cfg.apply_z_reflect_padding
    [raw_data, padInfo] = reflect_pad_raw_xyz_local(raw_data, cfg.fastSI, cfg.nphases, cfg.ndirs, ...
        cfg.y_reflect_pad_pixels, cfg.x_reflect_pad_pixels, cfg.z_reflect_pad_planes);
    fprintf(fid, 'Applied reflect padding: y=%d px, x=%d px, z=%d planes on each side\n', ...
        padInfo.padY, padInfo.padX, padInfo.padZ);
else
    padInfo = struct('used', false, 'padY', 0, 'padX', 0, 'padZ', 0);
end

% ----- Global refit on the cropped/raw channel data -----
fit_params = struct();
fit_params.nphases = cfg.nphases;
fit_params.ndirs = cfg.ndirs;
fit_params.pxl_dim_data = cfg.pxl_dim_data;
fit_params.k0angles = cfg.k0angles;
fit_params.linespacing = cfg.linespacing;
fit_params.background = background_for_fit_and_recon;
fit_params.fastSI = cfg.fastSI;
fit_params.otf_support_thresh = 0.006;
fit_params.refit_phase_range_deg = cfg.refit_phase_range_deg;
fit_params.refit_phase_step_deg = cfg.refit_phase_step_deg;
fit_params.refit_k0_search_window_px = cfg.refit_k0_search_window_px;

refit_fun = str2func(cfg.refit_function_name);
fitReport = feval(refit_fun, raw_data, fit_params);

k0angles_refined      = double(fitReport.k0angles_refined(:)).';
linespacing_refined   = double(fitReport.linespacing_refined(:)).';
phase_vectors_refined = double(fitReport.phase_vectors_rad);

proposed_phase_vectors_rad = build_proposed_phase_vectors(cfg.nphases, cfg.ndirs, cfg.phase_step_nm, cfg.linespacing);
proposed_phase_step_rad = effective_phase_step_rad(proposed_phase_vectors_rad);
refined_phase_step_rad  = effective_phase_step_rad(phase_vectors_refined);

angle_dev_deg = abs(rad2deg(wrap_to_pi(k0angles_refined - cfg.k0angles)));
angle_dev_pct = 100 * angle_dev_deg / 180;
linespacing_dev_pct = 100 * abs((linespacing_refined - cfg.linespacing) ./ max(abs(cfg.linespacing), eps));
phase_step_dev_pct  = 100 * abs((refined_phase_step_rad - proposed_phase_step_rad) ./ max(abs(proposed_phase_step_rad), eps));

if cfg.auto_accept_refit_if_within_threshold
    accept_angle       = angle_dev_pct <= cfg.refit_accept_threshold_pct;
    accept_linespacing = linespacing_dev_pct <= cfg.refit_accept_threshold_pct;
    accept_phase       = phase_step_dev_pct <= cfg.refit_accept_threshold_pct;
else
    accept_angle       = true(1, cfg.ndirs);
    accept_linespacing = true(1, cfg.ndirs);
    accept_phase       = true(1, cfg.ndirs);
end

k0angles_accepted = cfg.k0angles;
k0angles_accepted(accept_angle) = k0angles_refined(accept_angle);

linespacing_accepted = repmat(cfg.linespacing, 1, cfg.ndirs);
linespacing_accepted(accept_linespacing) = linespacing_refined(accept_linespacing);

phase_vectors_accepted = proposed_phase_vectors_rad;
phase_vectors_accepted(accept_phase, :) = phase_vectors_refined(accept_phase, :);

accepted_phase_step_rad = effective_phase_step_rad(phase_vectors_accepted);
accepted_phase_step_nm  = accepted_phase_step_rad ./ (2*pi) .* linespacing_accepted * 1e3;

fprintf(fid, 'Accepted angle (deg): %s\n', mat2str(rad2deg(k0angles_accepted), 6));
fprintf(fid, 'Accepted line spacing (um): %s\n', mat2str(linespacing_accepted, 6));
fprintf(fid, 'Accepted phase step (deg): %s\n', mat2str(rad2deg(accepted_phase_step_rad), 6));

% ----- OTF load or generate -----
otf_meta = struct();
if exist(job.output_otf, 'file') == 2 && ~cfg.overwrite_otf
    S = load(job.output_otf);
    assert(isfield(S, 'otf_data'), 'Cached OTF MAT missing variable otf_data');
    otf_data = S.otf_data;
    if isfield(S, 'otf_meta'), otf_meta = S.otf_meta; end
    fprintf(fid, 'Loaded cached OTF: %s\n', job.output_otf);
else
    psf_stack = readtiff(job.psf_file);
    psf_npages = size(psf_stack, 3);
    assert(mod(psf_npages, cfg.nphases * cfg.psf_ndirs) == 0, 'PSF TIFF page count incompatible with nphases*psf_ndirs');
    psf_nz = psf_npages / (cfg.nphases * cfg.psf_ndirs);
    psf_5d = reshape(psf_stack, size(psf_stack,1), size(psf_stack,2), cfg.nphases, psf_nz, cfg.psf_ndirs);
    psf_5d = permute(psf_5d, [1 2 4 3 5]);
    if cfg.psf_ndirs == 1
        psf_data = psf_5d(:,:,:,:,1);
    else
        psf_data = psf_5d;
    end
    clear psf_stack psf_5d;

    otf_params = struct();
    otf_params.pxl_dim_psf = cfg.pxl_dim_psf;
    otf_params.linespacing = linespacing_accepted(1);
    otf_params.k0angles = k0angles_accepted(1:cfg.psf_ndirs);
    otf_params.nphases = cfg.nphases;
    otf_params.background = cfg.background_psf;
    otf_params.auto_background = cfg.auto_background;
    otf_params.border_size = cfg.border_size;
    otf_params.napodize = cfg.napodize;
    otf_params.use_centering = true;
    otf_params.phase_step = [];
    otf_params.phase_vector_rad = phase_vectors_accepted(1:cfg.psf_ndirs,:);
    otf_params.bead_diameter = cfg.bead_diameter;
    otf_params.min_bead_tf = 0.02;
    otf_params.cleanup_threshold = cfg.cleanup_threshold;
    otf_params.tile_to_ndirs = false;
    otf_params.truncate_bead_tf_after_first_zero = cfg.truncate_bead_tf_after_first_zero;
    otf_params.wavelength = cfg.wavelength;
    otf_params.na = cfg.na;
    otf_params.nimm = cfg.nimm;
    otf_params.cleanup_otf = cfg.otf_cleanup_enable;
    otf_params.cylindrical_average = cfg.otf_cylindrical_average;
    otf_params.fixorigin = cfg.otf_fixorigin;
    otf_params.leavekz = cfg.otf_leavekz;
    otf_params.krmax = cfg.otf_krmax;
    otf_params.nocompen = cfg.otf_nocompen;
    otf_params.cleanup_interp_method = cfg.otf_cleanup_interp_method;
    otf_params.return_cartesian_preview = cfg.return_cartesian_preview;
    otf_params.force_polar_output = cfg.force_polar_output;
    otf_params.output_file = job.output_otf;

    otf_fun = str2func(cfg.otf_function_name);
    [otf_data, otf_meta] = feval(otf_fun, psf_data, otf_params);
    clear psf_data;
end

% ----- Reconstruction -----
recon_params = struct();
recon_params.nphases       = cfg.nphases;
recon_params.ndirs         = cfg.ndirs;
recon_params.pxl_dim_data  = cfg.pxl_dim_data;
recon_params.pxl_dim_psf   = cfg.pxl_dim_psf;
recon_params.wavelength    = cfg.wavelength;
recon_params.na            = cfg.na;
recon_params.nimm          = cfg.nimm;
recon_params.k0angles      = k0angles_accepted;
recon_params.linespacing   = mean(linespacing_accepted);
recon_params.phase_step    = [];
recon_params.phase_vector_rad = phase_vectors_accepted;
recon_params.wiener        = job.channel_wiener;
recon_params.background    = background_for_fit_and_recon;
recon_params.zoomfact      = cfg.zoomfact;
recon_params.otfRA         = (otf_num_dirs_local(otf_data) == 1);
recon_params.fastSI        = cfg.fastSI;
recon_params.gammaApo      = cfg.gammaApo;
recon_params.dampenOrder0  = cfg.dampenOrder0;
recon_params.suppress_dc   = cfg.suppress_dc;
recon_params.usePerDecomp  = cfg.usePerDecomp;
recon_params.k0_search     = cfg.k0_search;
recon_params.useGPU        = cfg.useGPU;
recon_params.debug         = cfg.debug;
recon_params.modamp_thresh = cfg.modamp_thresh;
recon_params.forcemodamp   = cfg.forcemodamp;
recon_params.suppress_band_centers = cfg.suppress_band_centers;
recon_params.band_suppress_radius_px = cfg.band_suppress_radius_px;
recon_params.band_suppress_min_weight = cfg.band_suppress_min_weight;
recon_params.band_suppress_power = cfg.band_suppress_power;

recon_params.use_tiling    = cfg.use_tiling;
recon_params.tile_size_xyz = cfg.tile_size_xyz;
recon_params.tile_overlap_xyz = cfg.tile_overlap_xyz;
recon_params.tile_refit    = cfg.tile_refit;
recon_params.tile_refit_accept_threshold_pct = cfg.tile_refit_accept_threshold_pct;
recon_params.tile_auto_accept_refit_if_within_threshold = cfg.tile_auto_accept_refit_if_within_threshold;
recon_params.tile_local_k0_search = cfg.tile_local_k0_search;
recon_params.tile_refit_score_min = cfg.tile_refit_score_min;
recon_params.tile_texture_cv_min  = cfg.tile_texture_cv_min;
recon_params.tile_parallel_refit = cfg.tile_parallel_refit;
recon_params.tile_parallel_recon = cfg.tile_parallel_recon;
recon_params.tile_parallel_workers = cfg.tile_parallel_workers;
recon_params.tile_multigpu = cfg.tile_multigpu;
recon_params.tile_single_gpu_batch = cfg.tile_single_gpu_batch;
recon_params.tile_gpu_batch_size = cfg.tile_gpu_batch_size;
recon_params.tile_batch_min_group = cfg.tile_batch_min_group;
recon_params.tile_disable_forcemodamp_on_low_texture = cfg.tile_disable_forcemodamp_on_low_texture;
recon_params.refit_k0_search_window_px = cfg.refit_k0_search_window_px;
recon_params.otf_support_thresh = 0.006;
recon_params.sideband_zcut_factor = 1.3;
recon_params.apodizeoutput = 1;
recon_params.amp_in_wiener = true;

if fileLevelParallelGPU
    recon_params.tile_parallel_refit = false;
    recon_params.tile_parallel_recon = false;
end

recon_fun = str2func(cfg.recon_function_name);
[recon, tileReport] = feval(recon_fun, raw_data, otf_data, recon_params);

if padInfo.used
    recon = crop_recon_xyz_local(recon, padInfo, cfg.zoomfact);
end

if cfg.save16bit
    recon(recon<0) = 0;
    writetiff(uint16(recon), job.output_recon);
else
    writetiff(recon, job.output_recon);
end

acceptanceReport = struct();
acceptanceReport.threshold_pct = cfg.refit_accept_threshold_pct;
acceptanceReport.auto_accept = cfg.auto_accept_refit_if_within_threshold;
acceptanceReport.k0angles_proposed = cfg.k0angles;
acceptanceReport.k0angles_refined = k0angles_refined;
acceptanceReport.k0angles_accepted = k0angles_accepted;
acceptanceReport.angle_dev_deg = angle_dev_deg;
acceptanceReport.angle_dev_pct = angle_dev_pct;
acceptanceReport.accept_angle = accept_angle;
acceptanceReport.linespacing_proposed = repmat(cfg.linespacing, 1, cfg.ndirs);
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
acceptanceReport.cropInfo = cropInfo;
acceptanceReport.channelCorrectionInfo = corrInfo;
acceptanceReport.channelWiener = job.channel_wiener;
acceptanceReport.bgInfo = bgInfo;
acceptanceReport.padInfo = padInfo;
acceptanceReport.zPadInfo = padInfo; % compatibility alias

% save(job.output_report, 'fitReport', 'acceptanceReport', 'recon_params', 'otf_meta', 'tileReport', '-v7.3');
% 
% fig = figure('Color','w','Visible','off');
% imagesc(max(recon,[],3)); axis image off; colormap gray; colorbar;
% title(sprintf('Reconstruction max projection: %s', job.raw_name), 'Interpreter', 'none');
% saveas(fig, job.output_maxproj_png); close(fig);
% 
% if cfg.use_tiling && isfield(tileReport,'used_tiling') && tileReport.used_tiling && cfg.save_tile_diagnostic_figures_flag
%     save_tile_diagnostic_figures(tileReport, job.tile_fig_prefix);
%     if isfield(tileReport, 'volume_fields')
%         save_volume_diagnostic_figures(tileReport.volume_fields, job.tile_fig_prefix);
%     end
% end
% 
% if cfg.save_fft_diagnostic_figure_flag
%     recon_voxel_xyz = [cfg.pxl_dim_data(1)/cfg.zoomfact, cfg.pxl_dim_data(2)/cfg.zoomfact, cfg.pxl_dim_data(3)];
%     save_fft_diagnostic_figure(recon, recon_voxel_xyz, job.output_fft_png, ...
%         cfg.fft_diag_resample_isotropic, cfg.fft_diag_iso_voxel_um);
% end
fprintf(fid, 'Finished channel/job successfully.\n');
end

toc
%%
function job = build_channel_job(raw_name, ch, chdir, cfg)
job = struct();
job.raw_name = raw_name;
job.channel_name = ch.name;
job.psf_file = resolve_path_local(ch.psf_file, cfg.script_dir);
job.output_recon = fullfile(chdir, raw_name);
[~, stem, ~] = fileparts(raw_name);
[~, stem_otf, ~] = fileparts(ch.psf_file);
job.output_otf = fullfile(chdir, [stem_otf, '_otf.mat']);
job.output_report = fullfile(chdir, [stem, '_report.mat']);
job.output_fft_png = fullfile(chdir, [stem, '_fft_planes.png']);
job.output_maxproj_png = fullfile(chdir, [stem, '_max_projection.png']);
job.tile_fig_prefix = fullfile(chdir, [stem, '_tilediag']);
job.channel_background = resolve_channel_scalar_setting(ch, 'background', cfg.background);
job.channel_wiener = resolve_channel_scalar_setting(ch, 'wiener', cfg.wiener);
end

function job = build_default_job(raw_name, cfg)
job = struct();
job.raw_name = raw_name;
job.channel_name = 'full';
job.psf_file = resolve_psf_file(cfg.psf_input, raw_name);
job.output_recon = fullfile(cfg.output_root, raw_name);
[~, stem, ~] = fileparts(raw_name);
[~, stem_otf, ~] = fileparts(job.psf_file);
job.output_otf = fullfile(cfg.output_root, [stem_otf, '_otf.mat']);
job.output_report = fullfile(cfg.output_root, [stem, '_report.mat']);
job.output_fft_png = fullfile(cfg.output_root, [stem, '_fft_planes.png']);
job.output_maxproj_png = fullfile(cfg.output_root, [stem, '_max_projection.png']);
job.tile_fig_prefix = fullfile(cfg.output_root, [stem, '_tilediag']);
job.channel_background = cfg.background;
job.channel_wiener = cfg.wiener;
end

function [raw_crop, info] = crop_raw_channel(raw_data, ch, nphases, ndirs, fastSI)
[ny, nx, nimgs] = size(raw_data);
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1) == 0, 'raw_data size incompatible with nphases*ndirs');
nz = round(nz);

yr = sanitize_range(getfield_or(ch, 'y_range', []), ny); %#ok<GFLD>
xr = sanitize_range(getfield_or(ch, 'x_range', []), nx); %#ok<GFLD>
zr = sanitize_range(getfield_or(ch, 'z_range', []), nz); %#ok<GFLD>

data5d = raw3d_to_5d_local(raw_data, fastSI, nphases, ndirs);
data5d = data5d(yr(1):yr(2), xr(1):xr(2), zr(1):zr(2), :, :);
raw_crop = raw5d_to_3d_local(data5d, fastSI);

info = struct();
info.used_crop = true;
info.y1 = yr(1); info.y2 = yr(2);
info.x1 = xr(1); info.x2 = xr(2);
info.z1 = zr(1); info.z2 = zr(2);
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

function r = sanitize_range(r, N)
if isempty(r)
    r = [1 N];
else
    r = round(double(r(:).'));
    assert(numel(r) == 2, 'Range must be empty or [start end].');
    r(1) = max(1, min(N, r(1)));
    r(2) = max(1, min(N, r(2)));
    assert(r(2) >= r(1), 'Invalid crop range [%d %d].', r(1), r(2));
end
end

function v = getfield_or(s, fname, defaultVal)
if isfield(s, fname)
    v = s.(fname);
else
    v = defaultVal;
end
end

function v = resolve_channel_scalar_setting(s, fname, globalDefault)
v = getfield_or(s, fname, []);
if isempty(v)
    v = globalDefault;
end
end

function p = resolve_path_local(p, script_dir)
if isempty(p)
    return;
end
if exist(p, 'file') == 2
    return;
end
cand = fullfile(script_dir, p);
if exist(cand, 'file') == 2
    p = cand;
end
end

function name = resolve_first_available(nameList)
name = '';
for i = 1:numel(nameList)
    if exist(nameList{i}, 'file') == 2
        name = nameList{i};
        return;
    end
end
error('None of the candidate function names were found on the MATLAB path.');
end

function safe_fclose(fid)
try
    if ~isempty(fid) && fid > 0
        fclose(fid);
    end
catch
end
end

function files = list_tiff_files(folderPath)
a = dir(fullfile(folderPath, '*.tif'));
b = dir(fullfile(folderPath, '*.tiff'));
files = [a; b];
files = files(~[files.isdir]);
end

function psf_file = resolve_psf_file(psf_input, raw_file_name)
if exist(psf_input, 'file') == 2
    psf_file = psf_input;
    return;
end

psf_files = list_tiff_files(psf_input);
assert(~isempty(psf_files), 'No TIFF files found in PSF folder.');

if numel(psf_files) == 1
    psf_file = fullfile(psf_files(1).folder, psf_files(1).name);
    return;
end

[~, raw_stem, ~] = fileparts(raw_file_name);
match_idx = [];
for i = 1:numel(psf_files)
    [~, psf_stem, ~] = fileparts(psf_files(i).name);
    if strcmpi(psf_stem, raw_stem)
        match_idx = i;
        break;
    end
end

assert(~isempty(match_idx), 'Could not resolve a PSF file for raw file %s.', raw_file_name);
psf_file = fullfile(psf_files(match_idx).folder, psf_files(match_idx).name);
end

function tf = can_use_parfor()
tf = ~isempty(ver('parallel')) && license('test','Distrib_Computing_Toolbox');
end

function pool = ensure_process_pool(numWorkers)
pool = gcp('nocreate');
if isempty(pool)
    if isempty(numWorkers)
        pool = parpool('local');
    else
        pool = parpool('local', numWorkers);
    end
end
end

function pool = ensure_gpu_process_pool(numWorkers, numThreads)
pool = gcp('nocreate');
if ~isempty(pool)
    if isa(pool, 'parallel.ProcessPool') && pool.NumWorkers == numWorkers
        return;
    else
        delete(pool);
        pool = [];
    end
end

try
    c = parcluster('Processes');
catch
    c = parcluster('local');
end

if isempty(numThreads)
    try
        totalWorkerThreads = double(c.NumWorkers);
    catch
        totalWorkerThreads = [];
    end
    if isempty(totalWorkerThreads) || totalWorkerThreads <= 0
        try
            totalWorkerThreads = maxNumCompThreads;
        catch
            totalWorkerThreads = numWorkers;
        end
    end
    numThreads = max(1, floor(totalWorkerThreads / max(1, numWorkers)));
end

try
    c.NumThreads = max(1, round(numThreads));
catch
end

pool = parpool(c, numWorkers);
end

function cfg = prepare_shared_channel_otfs_once(cfg)
fprintf('Preparing shared OTFs once up front...\n');

if cfg.use_channel_crops
    enabled = find([cfg.channel_configs.enabled]);
    for ii = 1:numel(enabled)
        ch = cfg.channel_configs(enabled(ii));
        chdir = fullfile(cfg.output_root, ch.name);
        if exist(chdir, 'dir') ~= 7
            mkdir(chdir);
        end
        shared_otf_file = shared_channel_otf_path(ch, chdir);
        if exist(shared_otf_file, 'file') == 2 && ~cfg.overwrite_otf
            fprintf('  Reusing shared OTF for %s: %s\n', ch.name, shared_otf_file);
            continue;
        end
        build_one_shared_otf(shared_otf_file, resolve_path_local(ch.psf_file, cfg.script_dir), cfg);
    end
else
    shared_otf_file = shared_default_otf_path(cfg);
    if ~(exist(shared_otf_file, 'file') == 2 && ~cfg.overwrite_otf)
        build_one_shared_otf(shared_otf_file, resolve_psf_file(cfg.psf_input, ''), cfg);
    end
end
end

function cfg = preload_global_camera_offset_once(cfg)
camFile = cfg.camera_offset_file;
if isempty(camFile)
    cfg.camera_offset_image = [];
    return;
end
fprintf('Preloading global camera offset once up front...\n');
camPath = resolve_path_local(camFile, cfg.script_dir);
camImg = readtiff(camPath);
if ndims(camImg) >= 3
    camImg = camImg(:,:,1);
end
cfg.camera_offset_file = camPath;
cfg.camera_offset_image = single(camImg);
fprintf('  Loaded global camera offset: %s\n', camPath);
end

function p = shared_channel_otf_path(ch, chdir)
[~, stem_otf, ~] = fileparts(ch.psf_file);
p = fullfile(chdir, [stem_otf, '_otf.mat']);
end

function p = shared_default_otf_path(cfg)
[~, stem_otf, ~] = fileparts(cfg.psf_input);
p = fullfile(cfg.output_root, [stem_otf, '_otf.mat']);
end

function build_one_shared_otf(output_otf_file, psf_file, cfg)
fprintf('  Building shared OTF: %s\n', output_otf_file);

psf_stack = readtiff(psf_file);
psf_npages = size(psf_stack, 3);
assert(mod(psf_npages, cfg.nphases * cfg.psf_ndirs) == 0, 'PSF TIFF page count incompatible with nphases*psf_ndirs');
psf_nz = psf_npages / (cfg.nphases * cfg.psf_ndirs);
psf_5d = reshape(psf_stack, size(psf_stack,1), size(psf_stack,2), cfg.nphases, psf_nz, cfg.psf_ndirs);
psf_5d = permute(psf_5d, [1 2 4 3 5]);
if cfg.psf_ndirs == 1
    psf_data = psf_5d(:,:,:,:,1);
else
    psf_data = psf_5d;
end
clear psf_stack psf_5d;

otf_params = struct();
otf_params.pxl_dim_psf = cfg.pxl_dim_psf;
otf_params.linespacing = cfg.linespacing;
otf_params.k0angles = cfg.k0angles(1:cfg.psf_ndirs);
otf_params.nphases = cfg.nphases;
otf_params.background = cfg.background_psf;
otf_params.auto_background = cfg.auto_background;
otf_params.border_size = cfg.border_size;
otf_params.napodize = cfg.napodize;
otf_params.use_centering = true;
otf_params.phase_step = [];
otf_params.phase_vector_rad = build_proposed_phase_vectors(cfg.nphases, cfg.ndirs, cfg.phase_step_nm, cfg.linespacing);
otf_params.phase_vector_rad = otf_params.phase_vector_rad(1:cfg.psf_ndirs,:);
otf_params.bead_diameter = cfg.bead_diameter;
otf_params.min_bead_tf = 0.02;
otf_params.cleanup_threshold = cfg.cleanup_threshold;
otf_params.tile_to_ndirs = false;
otf_params.truncate_bead_tf_after_first_zero = cfg.truncate_bead_tf_after_first_zero;
otf_params.wavelength = cfg.wavelength;
otf_params.na = cfg.na;
otf_params.nimm = cfg.nimm;
otf_params.cleanup_otf = cfg.otf_cleanup_enable;
otf_params.cylindrical_average = cfg.otf_cylindrical_average;
otf_params.fixorigin = cfg.otf_fixorigin;
otf_params.leavekz = cfg.otf_leavekz;
otf_params.krmax = cfg.otf_krmax;
otf_params.nocompen = cfg.otf_nocompen;
otf_params.cleanup_interp_method = cfg.otf_cleanup_interp_method;
otf_params.return_cartesian_preview = cfg.return_cartesian_preview;
otf_params.force_polar_output = cfg.force_polar_output;
otf_params.output_file = output_otf_file;

otf_fun = str2func(cfg.otf_function_name);
[~, ~] = feval(otf_fun, psf_data, otf_params);
clear psf_data;
end

function [raw_crop, info] = crop_raw_channel_from_5d_local(data5d, ch, fastSI)
[ny, nx, nz, ~, ~] = size(data5d);
yr = sanitize_range(getfield_or(ch, 'y_range', []), ny);
xr = sanitize_range(getfield_or(ch, 'x_range', []), nx);
zr = sanitize_range(getfield_or(ch, 'z_range', []), nz);

data5d_crop = data5d(yr(1):yr(2), xr(1):xr(2), zr(1):zr(2), :, :);
raw_crop = raw5d_to_3d_local(data5d_crop, fastSI);

info = struct();
info.used_crop = true;
info.y1 = yr(1); info.y2 = yr(2);
info.x1 = xr(1); info.x2 = xr(2);
info.z1 = zr(1); info.z2 = zr(2);
end

function [raw_corr, info] = apply_channel_corrections_local(raw_data, scalar_background)
raw_corr = single(raw_data);
info = struct('scalar_background', 0);

if isempty(scalar_background)
    scalar_background = 0;
end
raw_corr = raw_corr - single(scalar_background);
info.scalar_background = double(scalar_background);
end

function raw_corr = apply_global_camera_offset_local(raw_data, camera_offset_image)
raw_corr = single(raw_data);
img = single(camera_offset_image);
if ndims(img) >= 3
    img = img(:,:,1);
end
[nyi, nxi] = size(img);
[ny, nx, ~] = size(raw_corr);
if nyi ~= ny || nxi ~= nx
    error('Global camera offset image size [%d %d] must match full raw frame size [%d %d].', nxi, nyi, nx, ny);
end
raw_corr = bsxfun(@minus, raw_corr, img);
end

function n = available_gpu_count()
n = 0;
try
    if exist('gpuDeviceCount', 'file')
        try
            n = gpuDeviceCount('available');
        catch
            n = gpuDeviceCount;
        end
    end
catch
    n = 0;
end
end

function assign_worker_gpu(nGPUs)
try
    t = getCurrentTask();
    if isempty(t)
        idx = 1;
    else
        idx = mod(t.ID - 1, nGPUs) + 1;
    end
    gpuDevice(idx);
catch
end
end

function assign_main_gpu(idx)
try
    gpuDevice(idx);
catch
end
end

function v = min_nonempty(a, b)
if isempty(a)
    v = b;
else
    v = min(a, b);
end
end

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

% The next three figure helpers mirror the current runner behavior.

function save_tile_diagnostic_figures(tileReport, outPrefix)
if ~isfield(tileReport,'accepted_maps'), return; end
mapsA = tileReport.accepted_maps;
masks = tileReport.accept_masks;
score = tileReport.fit_score;
cvmap = tileReport.texture_cv;
refitUsed = tileReport.tile_refit_used;
[~, ~, nTz, ndirs] = size(mapsA.angle_deg);
midZ = max(1, round(nTz/2));

for d = 1:ndirs
    fig1 = figure('Color','w','Visible','off','Position',[100 100 1400 800]);
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
if ~isfield(volumeFields,'accepted'), return; end
mapsA = volumeFields.accepted;
mapsR = volumeFields.refined;
masks = volumeFields.accept_masks;
score = volumeFields.fit_score;
cvmap = volumeFields.texture_cv;
refitUsed = volumeFields.tile_refit_used;
[~, ~, nz, ndirs] = size(mapsA.angle_deg);
midZ = max(1, round(nz/2));

for d = 1:ndirs
    fig1 = figure('Color','w','Visible','off','Position',[100 100 1500 850]);
    tiledlayout(2,5,'Padding','compact','TileSpacing','compact');
    nexttile; imagesc(mean(mapsA.angle_deg(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Accepted angle z-mean dir %d', d));
    nexttile; imagesc(mean(mapsA.linespacing_um(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Accepted spacing z-mean dir %d', d));
    nexttile; imagesc(mean(mapsA.phase_step_deg(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Accepted phase z-mean dir %d', d));
    nexttile; imagesc(mean(score(:,:,:,d),3,'omitnan')); axis image; colorbar; title(sprintf('Fit score z-mean dir %d', d));
    nexttile; imagesc(mean(cvmap,3,'omitnan')); axis image; colorbar; title('Texture CV z-mean');
    nexttile; imagesc(mapsA.angle_deg(:,:,midZ,d)); axis image; colorbar; title(sprintf('Accepted angle z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsA.linespacing_um(:,:,midZ,d)); axis image; colorbar; title(sprintf('Accepted spacing z=%d dir %d', midZ, d));
    nexttile; imagesc(mapsA.phase_step_deg(:,:,midZ,d)); axis image; colorbar; title(sprintf('Accepted phase z=%d dir %d', midZ, d));
    nexttile; imagesc(score(:,:,midZ,d)); axis image; colorbar; title(sprintf('Fit score z=%d dir %d', midZ, d));
    nexttile; imagesc(double(refitUsed(:,:,midZ))); axis image; colorbar; title(sprintf('Tile refit used z=%d', midZ));
    saveas(fig1, sprintf('%s_volume_summary_dir%d.png', outPrefix, d));
    close(fig1);

    fig2 = figure('Color','w','Visible','off','Position',[100 100 1200 350]);
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imagesc(double(masks.angle(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept angle z=%d dir %d', midZ, d));
    nexttile; imagesc(double(masks.linespacing(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept spacing z=%d dir %d', midZ, d));
    nexttile; imagesc(double(masks.phase(:,:,midZ,d))); axis image; colorbar; title(sprintf('Accept phase z=%d dir %d', midZ, d));
    saveas(fig2, sprintf('%s_volume_accept_masks_dir%d.png', outPrefix, d));
    close(fig2);

    fig3 = figure('Color','w','Visible','off','Position',[100 100 1200 700]);
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

fig = figure('Color','w','Visible','off','Position',[100 100 1400 450]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(kx, ky, squeeze(Fshow(:,:,cz))); axis image; colorbar;
xlabel('k_x / k_{Nyq,x}'); ylabel('k_y / k_{Nyq,y}');
title('log_{10}(1+|F|/DC), k_x-k_y @ k_z=0');
nexttile;
imagesc(kx, kz, squeeze(Fshow(cy,:,:)).'); axis image; colorbar;
xlabel('k_x / k_{Nyq,x}'); ylabel('k_z / k_{Nyq,z}');
title('log_{10}(1+|F|/DC), k_x-k_z @ k_y=0');
nexttile;
imagesc(ky, kz, squeeze(Fshow(:,cx,:)).'); axis image; colorbar;
xlabel('k_y / k_{Nyq,y}'); ylabel('k_z / k_{Nyq,z}');
title('log_{10}(1+|F|/DC), k_y-k_z @ k_x=0');
saveas(fig, outFile);
close(fig);
end

function knull = normalized_k_axis(N, d)
idx = -ceil((N-1)/2):floor((N-1)/2);
k = idx / (N * d);
knyq = 0.5 / d;
knull = k / knyq;
end

function stem = strip_extension(fname)
[~, stem, ~] = fileparts(fname);
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


function [raw_corr, bgInfo] = subtract_spatial_background_local(raw_data, fastSI, nphases, ndirs, ...
    mode, open_radius_xy_px, smooth_z_sigma_px, gaussian_sigma_xy_px, gaussian_sigma_z_px)
mode = lower(string(mode));

if mode == "none"
    raw_corr = raw_data;
    bgInfo = struct('used', false, 'mode', 'none');
    return;
end

data5d = raw3d_to_5d_local(raw_data, fastSI, nphases, ndirs);
data5d = single(data5d);
[ny, nx, nz, ~, ndirs_in] = size(data5d);

bgVol = zeros(ny, nx, nz, ndirs_in, 'single');

for idir = 1:ndirs_in
    meanVol = mean(data5d(:,:,:,:,idir), 4);

    switch mode
        case "opening2d_meanphase"
            assert(exist('imopen', 'file') == 2 && exist('strel', 'file') == 2, ...
                'Morphological opening mode requires Image Processing Toolbox functions imopen/strel.');
            se = strel('disk', max(1, round(open_radius_xy_px)), 0);
            bg = zeros(size(meanVol), 'single');
            for iz = 1:size(meanVol,3)
                bg(:,:,iz) = single(imopen(meanVol(:,:,iz), se));
            end
            if smooth_z_sigma_px > 0
                bg = smooth_along_z_reflect_local(bg, smooth_z_sigma_px);
            end

        case "gaussian3d_meanphase"
            if exist('imgaussfilt3', 'file') == 2
                bg = single(imgaussfilt3(meanVol, [gaussian_sigma_xy_px, gaussian_sigma_xy_px, gaussian_sigma_z_px], ...
                    'Padding', 'symmetric'));
            else
                bg = separable_gaussian3d_reflect_local(meanVol, gaussian_sigma_xy_px, gaussian_sigma_z_px);
            end

        otherwise
            error('Unknown background_mode: %s', mode);
    end

    bgVol(:,:,:,idir) = bg;
end

for idir = 1:ndirs_in
    for iph = 1:nphases
        data5d(:,:,:,iph,idir) = data5d(:,:,:,iph,idir) - bgVol(:,:,:,idir);
    end
end

raw_corr = raw5d_to_3d_local(data5d, fastSI);
bgInfo = struct('used', true, 'mode', char(mode), ...
    'open_radius_xy_px', open_radius_xy_px, ...
    'smooth_z_sigma_px', smooth_z_sigma_px, ...
    'gaussian_sigma_xy_px', gaussian_sigma_xy_px, ...
    'gaussian_sigma_z_px', gaussian_sigma_z_px);
end

function vol = smooth_along_z_reflect_local(vol, sigma_z_px)
sigma_z_px = double(sigma_z_px);
if sigma_z_px <= 0
    return;
end
rad = max(1, ceil(3*sigma_z_px));
k = exp(-0.5 * ((-rad:rad) / sigma_z_px).^2);
k = single(k / sum(k));

pre  = vol(:,:,rad:-1:1);
post = vol(:,:,end:-1:end-rad+1);
pad = cat(3, pre, vol, post);
pad = convn(pad, reshape(k, [1 1 numel(k)]), 'same');
vol = pad(:,:,rad+1:end-rad);
end

function vol = separable_gaussian3d_reflect_local(vol, sigma_xy_px, sigma_z_px)
vol = smooth_along_xy_reflect_local(vol, sigma_xy_px);
vol = smooth_along_z_reflect_local(vol, sigma_z_px);
end

function vol = smooth_along_xy_reflect_local(vol, sigma_xy_px)
sigma_xy_px = double(sigma_xy_px);
if sigma_xy_px <= 0
    return;
end
rad = max(1, ceil(3*sigma_xy_px));
k = exp(-0.5 * ((-rad:rad) / sigma_xy_px).^2);
k = single(k / sum(k));

% smooth along y
preY  = vol(rad:-1:1,:,:);
postY = vol(end:-1:end-rad+1,:,:);
padY = cat(1, preY, vol, postY);
padY = convn(padY, reshape(k, [numel(k) 1 1]), 'same');
vol = padY(rad+1:end-rad,:,:);

% smooth along x
preX  = vol(:,rad:-1:1,:);
postX = vol(:,end:-1:end-rad+1,:);
padX = cat(2, preX, vol, postX);
padX = convn(padX, reshape(k, [1 numel(k) 1]), 'same');
vol = padX(:,rad+1:end-rad,:);
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
