function [recon] = sim_recon_2beam_3d_core(raw_data, otf_data, params)
% SIM_RECON_2BEAM_3D  GPU-accelerated 3D SIM reconstruction for 2-beam, 3-phase data.
%
% Compatible with make_otf_2beam_3d output:
%   otf_data(y, x, z, order, dir)
%   order = 1 -> order-0 / widefield OTF
%   order = 2 -> order+1 / first-order OTF
%
% Fixes relative to the previously shared version:
%   - CPU path no longer calls gather() unconditionally
%   - fftshift3/ifftshift3 use sequential dimension shifts for broader MATLAB compatibility
%   - debug_data storage restored
%   - stale clear of otf0_eff removed
%   - overlap-equalized modulation estimator made more numerically stable
%   - optional shifted-band-center suppression added to reduce ringing from
%     sharp DC-like peaks at the shifted order centers

%% ===== Parse parameters =====
required_fields = {'nphases','ndirs','pxl_dim_data','pxl_dim_psf','wavelength', ...
    'na','nimm','k0angles','linespacing','wiener'};
for k = 1:numel(required_fields)
    assert(isfield(params, required_fields{k}), 'Missing params.%s', required_fields{k});
end

nphases = params.nphases;
ndirs   = params.ndirs;
norders = 2;

assert(nphases == 3, 'This code is for 2-beam SIM: nphases must be 3.');
assert(numel(params.k0angles) == ndirs, 'numel(params.k0angles) must equal params.ndirs');

if ~isfield(params, 'background'),          params.background = 0;        end
if ~isfield(params, 'zoomfact'),            params.zoomfact = 2;          end
if ~isfield(params, 'otfRA'),               params.otfRA = true;          end
if ~isfield(params, 'fastSI'),              params.fastSI = false;        end
if ~isfield(params, 'gammaApo'),            params.gammaApo = 1;          end
if ~isfield(params, 'dampenOrder0'),        params.dampenOrder0 = false;  end
if ~isfield(params, 'suppress_dc'),         params.suppress_dc = true;    end
if ~isfield(params, 'usePerDecomp'),        params.usePerDecomp = true;   end
if ~isfield(params, 'k0_search'),           params.k0_search = true;      end
if ~isfield(params, 'debug'),               params.debug = false;         end
if ~isfield(params, 'phase_step'),          params.phase_step = [];       end
if ~isfield(params, 'phase_vector_rad'),    params.phase_vector_rad = []; end
if ~isfield(params, 'useGPU'),              params.useGPU = [];           end
if ~isfield(params, 'modamp_thresh'),       params.modamp_thresh = 0.05;  end
if ~isfield(params, 'forcemodamp'),         params.forcemodamp = [];       end
if ~isfield(params, 'pre_resampled_otf'),    params.pre_resampled_otf = false; end
if ~isfield(params, 'otf_support_thresh'),  params.otf_support_thresh = 0.006; end
if ~isfield(params, 'sep_cond_warn'),       params.sep_cond_warn = 1e4;   end
if ~isfield(params, 'suppress_band_centers'), params.suppress_band_centers = false; end
if ~isfield(params, 'band_suppress_radius_px'), params.band_suppress_radius_px = 6; end
if ~isfield(params, 'band_suppress_min_weight'), params.band_suppress_min_weight = 0.05; end
if ~isfield(params, 'band_suppress_power'), params.band_suppress_power = 6; end
if ~isfield(params, 'use_exact_otf_support'), params.use_exact_otf_support = true; end
if ~isfield(params, 'sideband_zcut_factor'), params.sideband_zcut_factor = 1.3; end
if ~isfield(params, 'apodizeoutput'), params.apodizeoutput = 1; end
if ~isfield(params, 'amp_in_wiener'), params.amp_in_wiener = true; end

wiener   = single(params.wiener);
zoomfact = params.zoomfact;

%% ===== GPU detection =====
useGPU = resolve_gpu_usage(params.useGPU);
if useGPU
    g = gpuDevice();
    fprintf('GPU acceleration: %s (%.1f GB free)\n', g.Name, g.AvailableMemory/1e9);
    toDevice = @(x) gpuArray(single(x));
else
    fprintf('GPU not available — running on CPU\n');
    toDevice = @(x) single(x);
end

%% ===== Determine data dimensions =====
[ny_raw, nx_raw, nimgs] = size(raw_data);
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1) == 0, 'Total images (%d) not divisible by nphases*ndirs (%d).', nimgs, nphases*ndirs);
nz = round(nz);
ny_out = ny_raw * zoomfact;
nx_out = nx_raw * zoomfact;

fprintf('Data: %d x %d x %d, %d dirs, %d phases\n', ny_raw, nx_raw, nz, ndirs, nphases);

%% ===== Background subtraction and vectorized reorganization =====
raw_data = single(raw_data) - single(params.background);
% Do NOT clamp negatives; it biases the phase separation and modamp estimate.

if params.fastSI
    data_5d = reshape(raw_data, ny_raw, nx_raw, nphases, ndirs, nz);
    data_5d = permute(data_5d, [1 2 5 3 4]); % [ny nx nz phase dir]
else
    data_5d = reshape(raw_data, ny_raw, nx_raw, nphases, nz, ndirs);
    data_5d = permute(data_5d, [1 2 4 3 5]); % [ny nx nz phase dir]
end
clear raw_data;

%% ===== Resample OTF on CPU =====
if params.pre_resampled_otf
    OTF = otf_data;
else
    OTF = resample_otf_3d(otf_data, params.pxl_dim_psf, [ny_raw, nx_raw, nz], ...
        params.pxl_dim_data, norders, ndirs, params.otfRA);
end

%% ===== Build separation matrices =====
sep_inv_dev_all = cell(1, ndirs);
noiseWeightsDev_all = cell(1, ndirs);
for idir = 1:ndirs
    phase_vec = resolve_phase_vector_for_dir(params, idir);
    sep_matrix = make_sep_from_phases(phase_vec);
    sep_cond = cond(double(sep_matrix));
    if sep_cond > params.sep_cond_warn
        warning('sim_recon_2beam_3d:IllConditionedSeparation', ...
            'Direction %d separation matrix condition number is high (%.3g). Using pseudo-inverse.', idir, sep_cond);
        sep_inv = cast(pinv(double(sep_matrix)), 'like', sep_matrix);
    else
        sep_inv = sep_matrix \ eye(size(sep_matrix), 'like', sep_matrix);
    end
    fprintf('Direction %d separation matrix condition number: %.4f', idir, sep_cond);
    sep_inv_dev_all{idir} = toDevice(sep_inv);
    noiseVarFactors = compute_noise_variance_factors(sep_inv);
    noiseWeights = single(1 ./ max(noiseVarFactors, eps('single')));
    noiseWeightsDev_all{idir} = toDevice(reshape(noiseWeights, [1 1 1 numel(noiseWeights)]));
end

%% ===== Pre-build reusable grids =====
shift_grid = build_shift_coordinate_vectors(ny_out, nx_out, nz, useGPU);
if params.usePerDecomp
    perdenom = build_perdecomp_denom(ny_raw, nx_raw, nz, useGPU);
else
    perdenom = [];
end
if params.suppress_band_centers
    band_center_weight_base = build_band_center_suppression_mask( ...
        ny_out, nx_out, useGPU, ...
        params.band_suppress_radius_px, ...
        params.band_suppress_min_weight, ...
        params.band_suppress_power);
else
    band_center_weight_base = [];
end

cudaFilter = build_cuda_style_filter_masks(ny_out, nx_out, nz, params.pxl_dim_data, zoomfact, ...
    params.na, params.nimm, params.wavelength, params.linespacing, ...
    params.sideband_zcut_factor, params.apodizeoutput, useGPU);

%% ===== Allocate accumulators =====
numerator   = complex_zeros_like_device([ny_out, nx_out, nz], useGPU);
denominator = real_zeros_like_device([ny_out, nx_out, nz], useGPU);

% Cache rotationally-averaged OTFs in upsampled form.
if params.otfRA
    otf_base_dev = toDevice(OTF(:,:,:,:,1));
    otf0_up_base = upsample_band_3d_gpu(otf_base_dev(:,:,:,1), ny_out, nx_out);
    otf1_up_base = upsample_band_3d_gpu(otf_base_dev(:,:,:,2), ny_out, nx_out);
else
    otf0_up_base = [];
    otf1_up_base = [];
end

if params.debug
    debug_data = struct();
end

total_timer = tic;

for idir = 1:ndirs
    dir_timer = tic;
    fprintf('\n--- Direction %d/%d (angle = %.4f rad) ---\n', idir, ndirs, params.k0angles(idir));

    %% Step 1: upload one direction block and FFT each phase
    data_dir = toDevice(data_5d(:,:,:,:,idir)); % [ny nx nz phase]
    if params.usePerDecomp
        phase_ft = complex_zeros_like_device([ny_raw, nx_raw, nz, nphases], useGPU);
        for iph = 1:nphases
            vol = perdecomp_3d_apply(data_dir(:,:,:,iph), perdenom);
            phase_ft(:,:,:,iph) = fftshift3(fft3_spatial(vol));
        end
    else
        phase_ft = fftshift3(fft3_spatial(data_dir));
    end
    clear data_dir;

    %% Step 2: batch-separate orders
    bands = batch_separate(phase_ft, sep_inv_dev_all{idir});
    clear phase_ft;

    %% Step 3: k0 search (2D only because k0z = 0)
    if params.k0_search
        k0_pixels = find_k0_2d_from_zsum(bands, params.k0angles(idir), params.linespacing, ...
            params.pxl_dim_data, [ny_raw, nx_raw, nz]);
    else
        k0_pixels = compute_nominal_k0(params.k0angles(idir), params.linespacing, ...
            params.pxl_dim_data, [ny_raw, nx_raw, nz]);
    end
    fprintf('  k0 = [%.3f, %.3f, %.3f] pixels\n', k0_pixels(1), k0_pixels(2), k0_pixels(3));

    %% Step 4: OTF selection and one-time upsampling
    if params.otfRA
        otf0_up = otf0_up_base;
        otf1_up = otf1_up_base;
    else
        otf_dir = toDevice(OTF(:,:,:,:,idir));
        otf0_up = upsample_band_3d_gpu(otf_dir(:,:,:,1), ny_out, nx_out);
        otf1_up = upsample_band_3d_gpu(otf_dir(:,:,:,2), ny_out, nx_out);
        clear otf_dir;
    end

    %% Step 5: one-time upsampling and batched shifting
    band0_up   = upsample_band_3d_gpu(bands(:,:,:,2), ny_out, nx_out);
    band_p1_up = upsample_band_3d_gpu(bands(:,:,:,3), ny_out, nx_out);
    band_m1_up = upsample_band_3d_gpu(bands(:,:,:,1), ny_out, nx_out);
    clear bands;

    k0_up = [k0_pixels(1) * zoomfact, k0_pixels(2) * zoomfact, k0_pixels(3)];
    shift_pos = build_shift_cache(k0_up, shift_grid);
    shift_neg = build_conjugate_shift_cache(shift_pos);

    vols_pos = cat(4, band_p1_up, otf1_up);
    shifted_pos = apply_shift_cache_batch(vols_pos, shift_pos);
    band_p1_shifted = shifted_pos(:,:,:,1);
    otf1_shifted    = shifted_pos(:,:,:,2);
    clear vols_pos shifted_pos band_p1_up;

    vols_neg = cat(4, band_m1_up, conj(otf1_up));
    shifted_neg = apply_shift_cache_batch(vols_neg, shift_neg);
    band_m1_shifted = shifted_neg(:,:,:,1);
    otf_m1_shifted  = shifted_neg(:,:,:,2);
    clear vols_neg shifted_neg band_m1_up otf1_up;

    if params.suppress_band_centers
        center_weight_0  = band_center_weight_base;
        center_weight_p1 = apply_shift_cache_2d(band_center_weight_base, shift_pos);
        center_weight_m1 = apply_shift_cache_2d(band_center_weight_base, shift_neg);
    else
        center_weight_0  = single(1);
        center_weight_p1 = single(1);
        center_weight_m1 = single(1);
    end

    %% Step 6: overlap-equalized modulation amplitude on already-shifted data
    [mod_amp, mod_diag] = estimate_modamp_overlap_equalized(band0_up, band_p1_shifted, ...
        otf0_up, otf1_shifted, params.otf_support_thresh);
    fprintf('  Modulation amplitude: %.4f |xsy|=%.4g sx=%.4g sy=%.4g overlap=%d\n', ...
        abs(mod_amp), abs(mod_diag.xsy), mod_diag.sx, mod_diag.sy, mod_diag.n_mask);

    [mod_amp_eff, modamp_mode] = choose_modamp(mod_amp, idir, params);
    switch modamp_mode
        case 'forced'
            fprintf('  Using forcemodamp override: %.4f\n', abs(mod_amp_eff));
        case 'capped'
            fprintf('  Low estimated modulation amplitude (%.4f); capping Wiener amp weight at %.2fx\n', ...
                abs(mod_amp), 1 / params.modamp_thresh);
    end

    amp_mag2 = single(max(abs(mod_amp_eff).^2, eps('single')));
    amp_conj = single(conj(mod_amp_eff));
    amp_fwd  = single(mod_amp_eff);

    %% Step 7: Wiener assembly with separation-noise weighting, exact axial support, and CUDA-style amp weighting
    weight_m1 = noiseWeightsDev_all{idir}(:,:,:,1) .* center_weight_m1 .* cudaFilter.zmask1;
    weight_0  = noiseWeightsDev_all{idir}(:,:,:,2) .* center_weight_0  .* cudaFilter.zmask0;
    weight_p1 = noiseWeightsDev_all{idir}(:,:,:,3) .* center_weight_p1 .* cudaFilter.zmask1;

    if params.dampenOrder0
        weight_0 = weight_0 * single(0.1);
    end

    if params.amp_in_wiener
        numerator = numerator ...
            + weight_0  .* conj(otf0_up)        .* band0_up ...
            + weight_p1 .* amp_conj .* conj(otf1_shifted)   .* band_p1_shifted ...
            + weight_m1 .* amp_fwd  .* conj(otf_m1_shifted) .* band_m1_shifted;

        denominator = denominator ...
            + weight_0  .* abs(otf0_up).^2 ...
            + weight_p1 .* amp_mag2 .* abs(otf1_shifted).^2 ...
            + weight_m1 .* amp_mag2 .* abs(otf_m1_shifted).^2;
    else
        band_p1_shifted = band_p1_shifted / mod_amp_eff;
        band_m1_shifted = band_m1_shifted / conj(mod_amp_eff);

        numerator = numerator ...
            + weight_0  .* conj(otf0_up)        .* band0_up ...
            + weight_p1 .* conj(otf1_shifted)   .* band_p1_shifted ...
            + weight_m1 .* conj(otf_m1_shifted) .* band_m1_shifted;

        denominator = denominator ...
            + weight_0  .* abs(otf0_up).^2 ...
            + weight_p1 .* abs(otf1_shifted).^2 ...
            + weight_m1 .* abs(otf_m1_shifted).^2;
    end

    if params.debug
        debug_data.k0{idir} = maybe_gather(k0_pixels);
        debug_data.modamp{idir} = mod_amp;
    end

    clear band0_up band_p1_shifted band_m1_shifted otf0_up otf1_shifted otf_m1_shifted center_weight_0 center_weight_p1 center_weight_m1;
    fprintf('  Direction %d done in %.1f s\n', idir, toc(dir_timer));
end

%% ===== Wiener filter =====
fprintf('\nWiener filter (w = %.4f)...\n', wiener);
if params.suppress_dc
    cy = floor(ny_out/2) + 1;
    cx = floor(nx_out/2) + 1;
    cz = floor(nz/2) + 1;
    denominator(cy, cx, cz) = max(denominator(:)) * single(0.1);
end
recon_ft = numerator ./ (denominator + wiener^2);
clear numerator denominator;

%% ===== Apodization =====
if params.apodizeoutput ~= 0
    recon_ft = apply_cuda_style_apodization_3d_gpu(recon_ft, cudaFilter);
end

%% ===== Inverse FFT =====
recon = real(ifft3_spatial(ifftshift3(recon_ft)));
clear recon_ft;

if useGPU
    recon = gather(recon);
end
recon = single(recon);

fprintf('Reconstruction complete: [%d, %d, %d] in %.1f s total\n', ...
    size(recon,1), size(recon,2), size(recon,3), toc(total_timer));

if params.debug
    assignin('base', 'debug_sim', debug_data);
end
end

%% =====================================================================
%  HELPER FUNCTIONS
%  =====================================================================

function useGPU = resolve_gpu_usage(userSetting)
useGPU = false;
try
    if exist('gpuDeviceCount', 'file')
        try
            n = gpuDeviceCount('available');
        catch
            n = gpuDeviceCount;
        end
        useGPU = (n > 0);
    end
catch
    useGPU = false;
end
if ~isempty(userSetting)
    useGPU = logical(userSetting) && useGPU;
end
end

function z = real_zeros_like_device(sz, useGPU)
if useGPU
    z = gpuArray.zeros(sz, 'single');
else
    z = zeros(sz, 'single');
end
end

function z = complex_zeros_like_device(sz, useGPU)
if useGPU
    z = complex(gpuArray.zeros(sz, 'single'));
else
    z = complex(zeros(sz, 'single'));
end
end

function out = fft3_spatial(in)
out = fft(in, [], 1);
out = fft(out, [], 2);
out = fft(out, [], 3);
end

function out = ifft3_spatial(in)
out = ifft(in, [], 1);
out = ifft(out, [], 2);
out = ifft(out, [], 3);
end

function out = fftshift3(in)
out = fftshift(in, 1);
out = fftshift(out, 2);
out = fftshift(out, 3);
end

function out = ifftshift3(in)
out = ifftshift(in, 1);
out = ifftshift(out, 2);
out = ifftshift(out, 3);
end

function bands = batch_separate(phase_ft, sep_inv)
[ny, nx, nz, nph] = size(phase_ft);
P = reshape(phase_ft, [], nph);
B = P * sep_inv.';
bands = reshape(B, ny, nx, nz, nph);
end

function grid = build_shift_coordinate_vectors(ny, nx, nz, useGPU)
grid.y = single((0:ny-1)' / ny);
grid.x = single((0:nx-1)  / nx);
grid.z = single(reshape((0:nz-1) / nz, [1 1 nz]));
if useGPU
    grid.y = gpuArray(grid.y);
    grid.x = gpuArray(grid.x);
    grid.z = gpuArray(grid.z);
end
end

function cache = build_shift_cache(shift_pixels, grid)
cache.int_shift = round(shift_pixels);
frac_shift = shift_pixels - cache.int_shift;
cache.has_frac = any(abs(frac_shift) > 1e-6);
if cache.has_frac
    cache.phase_y = exp(1i * single(2*pi*frac_shift(1)) * grid.y);
    cache.phase_x = exp(1i * single(2*pi*frac_shift(2)) * grid.x);
    cache.phase_z = exp(1i * single(2*pi*frac_shift(3)) * grid.z);
else
    cache.phase_y = [];
    cache.phase_x = [];
    cache.phase_z = [];
end
end

function cache_neg = build_conjugate_shift_cache(cache_pos)
cache_neg.int_shift = -cache_pos.int_shift;
cache_neg.has_frac  = cache_pos.has_frac;
if cache_pos.has_frac
    cache_neg.phase_y = conj(cache_pos.phase_y);
    cache_neg.phase_x = conj(cache_pos.phase_x);
    cache_neg.phase_z = conj(cache_pos.phase_z);
else
    cache_neg.phase_y = [];
    cache_neg.phase_x = [];
    cache_neg.phase_z = [];
end
end

function shifted = apply_shift_cache_batch(vols, cache)
shifted = circshift(vols, [cache.int_shift 0]);
if cache.has_frac
    tmp = ifft3_spatial(ifftshift3(shifted));
    tmp = tmp .* cache.phase_y .* cache.phase_x .* cache.phase_z;
    shifted = fftshift3(fft3_spatial(tmp));
end
end

function shifted = apply_shift_cache_2d(mask2d, cache)
shifted = circshift(mask2d, cache.int_shift(1:2));
if cache.has_frac
    tmp = ifft2(ifftshift(ifftshift(shifted,1),2));
    tmp = tmp .* cache.phase_y .* cache.phase_x;
    shifted = fftshift(fftshift(fft2(tmp),1),2);
end
shifted = real(shifted);
end

function w = build_band_center_suppression_mask(ny, nx, useGPU, radius_px, min_weight, power_val)
if radius_px <= 0
    if useGPU
        w = gpuArray.ones(ny, nx, 'single');
    else
        w = ones(ny, nx, 'single');
    end
    return;
end
yy = single((-ceil((ny-1)/2):floor((ny-1)/2))');
xx = single((-ceil((nx-1)/2):floor((nx-1)/2)));
if useGPU
    yy = gpuArray(yy);
    xx = gpuArray(xx);
end
rr = sqrt(yy.^2 + xx.^2);
x = rr / single(max(radius_px, eps('single')));
w = single(min_weight) + single(1 - min_weight) .* (x.^single(power_val) ./ (1 + x.^single(power_val)));
w = min(single(1), max(single(min_weight), w));
end

function out = upsample_band_3d_gpu(band_in, ny_out, nx_out)
[ny_in, nx_in, nz] = size(band_in);
if ny_in == ny_out && nx_in == nx_out
    out = band_in;
    return;
end
if isa(band_in, 'gpuArray')
    out = complex(gpuArray.zeros(ny_out, nx_out, nz, 'single'));
else
    out = complex(zeros(ny_out, nx_out, nz, 'single'));
end
y_start = floor((ny_out - ny_in)/2) + 1;
x_start = floor((nx_out - nx_in)/2) + 1;
out(y_start:y_start+ny_in-1, x_start:x_start+nx_in-1, :) = band_in;
out = out * single((ny_out * nx_out) / (ny_in * nx_in));
end

function perdenom = build_perdecomp_denom(ny, nx, nz, useGPU)
ii = single((0:ny-1)');
jj = single((0:nx-1));
kk = single(reshape(0:nz-1, [1 1 nz]));
if useGPU
    ii = gpuArray(ii); jj = gpuArray(jj); kk = gpuArray(kk);
end
perdenom = 2*cos(2*pi*ii/ny) + 2*cos(2*pi*jj/nx) + 2*cos(2*pi*kk/nz) - 6;
perdenom(1,1,1) = 1;
end

function vol_out = perdecomp_3d_apply(vol_in, perdenom)
bnd = zeros(size(vol_in), 'like', vol_in);
bnd(1,:,:)   = bnd(1,:,:)   + vol_in(1,:,:) - vol_in(end,:,:);
bnd(end,:,:) = bnd(end,:,:) + vol_in(end,:,:) - vol_in(1,:,:);
bnd(:,1,:)   = bnd(:,1,:)   + vol_in(:,1,:) - vol_in(:,end,:);
bnd(:,end,:) = bnd(:,end,:) + vol_in(:,end,:) - vol_in(:,1,:);
bnd(:,:,1)   = bnd(:,:,1)   + vol_in(:,:,1) - vol_in(:,:,end);
bnd(:,:,end) = bnd(:,:,end) + vol_in(:,:,end) - vol_in(:,:,1);
smooth = real(ifft3_spatial(fft3_spatial(bnd) ./ perdenom));
smooth = smooth - mean(smooth(:));
vol_out = vol_in - smooth;
end

function sep = make_separation_matrix_2beam_equal(nphases)
orders = [-1, 0, 1];
sep = complex(zeros(nphases, numel(orders), 'single'));
for j = 1:nphases
    phi_j = single((j-1) * 2*pi / nphases);
    for m = 1:numel(orders)
        sep(j,m) = exp(1i * single(orders(m)) * phi_j);
    end
end
end

function sep = make_separation_matrix_2beam(nphases, phase_step_nm, linespacing_um)
delta_phi = single((phase_step_nm * 1e-3) / linespacing_um * 2 * pi);
orders = [-1, 0, 1];
sep = complex(zeros(nphases, numel(orders), 'single'));
for j = 1:nphases
    for m = 1:numel(orders)
        sep(j,m) = exp(1i * single(orders(m)) * single(j-1) * delta_phi);
    end
end
end

function phase_vec = resolve_phase_vector_for_dir(params, idir)
if ~isempty(params.phase_vector_rad)
    pv = params.phase_vector_rad;
    if isvector(pv)
        assert(numel(pv) == 3, 'params.phase_vector_rad must be 1x3 or ndirsx3');
        phase_vec = single(reshape(pv, 1, 3));
    else
        assert(size(pv,2) == 3 && size(pv,1) >= idir, ...
            'params.phase_vector_rad must be 1x3 or ndirsx3');
        phase_vec = single(pv(idir,:));
    end
elseif ~isempty(params.phase_step)
    delta_phi = single((params.phase_step * 1e-3) / params.linespacing * 2*pi);
    phase_vec = single([0, delta_phi, 2*delta_phi]);
else
    phase_vec = single([0, 2*pi/3, 4*pi/3]);
end
phase_vec = align_phase_vector_monotonic(phase_vec);
end

function phase_vec = align_phase_vector_monotonic(phase_vec)
phase_vec = single(phase_vec(:).');
phase_vec = phase_vec - phase_vec(1);
for j = 2:numel(phase_vec)
    while phase_vec(j) <= phase_vec(j-1)
        phase_vec(j) = phase_vec(j) + single(2*pi);
    end
end
end

function sep = make_sep_from_phases(phases_rad)
orders = [-1, 0, 1];
sep = complex(zeros(numel(phases_rad), numel(orders), 'single'));
for j = 1:numel(phases_rad)
    for m = 1:numel(orders)
        sep(j,m) = exp(1i * single(orders(m)) * single(phases_rad(j)));
    end
end
end

function noiseVarFactors = compute_noise_variance_factors(sep_inv)
noiseVarFactors = single(sum(abs(sep_inv).^2, 2)).';
end


function OTF_out = resample_otf_3d(otf_in, pxl_psf, data_dims, pxl_data, norders, ndirs, otfRA)
if otf_is_polar_struct(otf_in)
    OTF_out = resample_polar_otf_to_cartesian(otf_in, data_dims, pxl_data, norders, ndirs, otfRA);
    return;
end
[ny_otf, nx_otf, nz_otf, ~, ndirs_otf] = size(otf_in);
ny = data_dims(1); nx = data_dims(2); nz = data_dims(3);
dk_otf = 1 ./ ([ny_otf, nx_otf, nz_otf] .* pxl_psf);
otf_yy = (-ceil((ny_otf-1)/2):floor((ny_otf-1)/2)) * dk_otf(1);
otf_xx = (-ceil((nx_otf-1)/2):floor((nx_otf-1)/2)) * dk_otf(2);
otf_zz = (-ceil((nz_otf-1)/2):floor((nz_otf-1)/2)) * dk_otf(3);
dk_data = 1 ./ ([ny, nx, nz] .* pxl_data);
map_yy = (-ceil((ny-1)/2):floor((ny-1)/2)) * dk_data(1);
map_xx = (-ceil((nx-1)/2):floor((nx-1)/2)) * dk_data(2);
map_zz = (-ceil((nz-1)/2):floor((nz-1)/2)) * dk_data(3);
[OTF_XX, OTF_YY, OTF_ZZ] = meshgrid(otf_xx, otf_yy, otf_zz);
[MAP_XX, MAP_YY, MAP_ZZ] = meshgrid(map_xx, map_yy, map_zz);
n_dirs_out = 1;
if ~otfRA, n_dirs_out = ndirs; end
OTF_out = complex(zeros(ny, nx, nz, norders, n_dirs_out, 'single'));
fprintf('Resampling OTF to [%d, %d, %d]...', ny, nx, nz);
for d = 1:n_dirs_out
    for ord = 1:norders
        otf_vol = otf_in(:,:,:,ord,min(d,ndirs_otf));
        OTF_out(:,:,:,ord,d) = single(interp3(OTF_XX, OTF_YY, OTF_ZZ, ...
            otf_vol, MAP_XX, MAP_YY, MAP_ZZ, 'cubic', 0+0i));
        fprintf('.');
    end
end
fprintf(' done.\n');
end

function k0_pix = compute_nominal_k0(angle_rad, linespacing, pxl_dim, dims)
ny = dims(1); nx = dims(2);
k0_phys = 1 / linespacing;
dk_y = 1 / (ny * pxl_dim(1));
dk_x = 1 / (nx * pxl_dim(2));
k0_pix = [k0_phys * cos(angle_rad) / dk_y, ...
    k0_phys * sin(angle_rad) / dk_x, 0];
end

function k0_pix = find_k0_2d_from_zsum(bands, angle_rad, linespacing, pxl_dim, dims)
k0_nominal = compute_nominal_k0(angle_rad, linespacing, pxl_dim, dims);
band0 = bands(:,:,:,2);
band1 = bands(:,:,:,3);
spec_xy = sum(conj(band0) .* band1, 3);
spec_xy = ifftshift(spec_xy, 1);
spec_xy = ifftshift(spec_xy, 2);
xcorr_xy = ifft2(spec_xy);
xcorr_xy = fftshift(xcorr_xy, 1);
xcorr_xy = fftshift(xcorr_xy, 2);

[ny, nx] = size(xcorr_xy);
cy = floor(ny/2) + 1;
cx = floor(nx/2) + 1;
pk_y = round(k0_nominal(1)) + cy;
pk_x = round(k0_nominal(2)) + cx;
win = 5;
y_range = max(1, pk_y-win):min(ny, pk_y+win);
x_range = max(1, pk_x-win):min(nx, pk_x+win);
xcorr_sub = maybe_gather(abs(xcorr_xy(y_range, x_range)));
[~, idx] = max(xcorr_sub(:));
[iy, ix] = ind2sub(size(xcorr_sub), idx);
[dy, dx] = subpixel_refine_patch_2d(xcorr_sub, iy, ix);
k0_pix = [y_range(iy) + dy - cy, x_range(ix) + dx - cx, 0];
fprintf('  k0 search: nominal [%.2f, %.2f, 0] -> refined [%.3f, %.3f, 0]\n', ...
    k0_nominal(1), k0_nominal(2), k0_pix(1), k0_pix(2));
end

function [dy, dx] = subpixel_refine_patch_2d(patch, iy, ix)
dy = 0; dx = 0;
[ny, nx] = size(patch);
if iy > 1 && iy < ny
    ym1 = log(max(patch(iy-1, ix), eps('single')));
    y0  = log(max(patch(iy,   ix), eps('single')));
    yp1 = log(max(patch(iy+1, ix), eps('single')));
    den = ym1 - 2*y0 + yp1;
    if abs(den) > eps('single')
        dy = 0.5 * (ym1 - yp1) / den;
    end
end
if ix > 1 && ix < nx
    xm1 = log(max(patch(iy, ix-1), eps('single')));
    x0  = log(max(patch(iy, ix),   eps('single')));
    xp1 = log(max(patch(iy, ix+1), eps('single')));
    den = xm1 - 2*x0 + xp1;
    if abs(den) > eps('single')
        dx = 0.5 * (xm1 - xp1) / den;
    end
end
end

function [mod_amp, mod_diag] = estimate_modamp_overlap_equalized(band0_up, band1_shifted, otf0_up, otf1_shifted, supportThresh)
a0 = abs(otf0_up);
a1 = abs(otf1_shifted);
mask = (a0 > supportThresh * max(a0(:))) & (a1 > supportThresh * max(a1(:)));
n_mask = maybe_gather(sum(mask(:)));
mod_diag = struct('xsy', 0, 'sx', 0, 'sy', 0, 'n_mask', n_mask, 'beta', 0);
if n_mask < 10
    fprintf('  WARNING: very small overlap region (%d voxels)\n', n_mask);
    mod_amp = 1;
    return;
end

eqnorm = sqrt(a0.^2 + a1.^2 + eps('single'));
ov0 = band0_up      .* conj(otf1_shifted) ./ eqnorm;
ov1 = band1_shifted .* conj(otf0_up)      ./ eqnorm;

xsy = maybe_gather(sum(conj(ov0(mask)) .* ov1(mask)));
sx  = maybe_gather(sum(abs(ov0(mask)).^2));
sy  = maybe_gather(sum(abs(ov1(mask)).^2));
mod_diag.xsy = xsy;
mod_diag.sx = sx;
mod_diag.sy = sy;

if sx <= eps('single') || sy <= eps('single')
    mod_amp = 1;
    return;
end

beta = 0.5 * atan2(2 * abs(xsy), (sx - sy));
if beta < 0
    beta = beta + 0.5*pi;
end
mod_diag.beta = beta;
mod_amp = tan(beta) * exp(1i * angle(xsy));

mag = abs(mod_amp);
if mag <= eps('single')
    mod_amp = 1;
else
    mod_amp = mod_amp / mag * min(mag, 1.0);
end
end

function [mod_amp_eff, mode] = choose_modamp(mod_amp, idir, params)
phase_est = angle(mod_amp);
mag_est = abs(mod_amp);

if ~isempty(params.forcemodamp)
    fm = params.forcemodamp;
    if isscalar(fm)
        fm_val = fm;
    elseif numel(fm) >= idir
        fm_val = fm(idir);
    else
        error('params.forcemodamp must be scalar or have at least ndirs elements.');
    end

    if ~isreal(fm_val)
        mod_amp_eff = single(fm_val);
    else
        mod_amp_eff = single(max(abs(fm_val), eps('single')) * exp(1i * phase_est));
    end
    mode = 'forced';
    return;
end

mag_eff = max(mag_est, params.modamp_thresh);
mod_amp_eff = single(mag_eff * exp(1i * phase_est));
if mag_est < params.modamp_thresh
    mode = 'capped';
else
    mode = 'estimated';
end
end




function cudaFilter = build_cuda_style_filter_masks(ny, nx, nz, pxl_dim, zoomfact, na, nimm, wavelength, linespacing, sidebandFactor, apodizeoutput, useGPU)
dy = pxl_dim(1) / zoomfact;
dx = pxl_dim(2) / zoomfact;
dz = pxl_dim(3);

dk_y = 1 / (ny * dy);
dk_x = 1 / (nx * dx);
dk_z = 1 / (nz * dz);

yy = single((-ceil((ny-1)/2):floor((ny-1)/2))' * dk_y);
xx = single((-ceil((nx-1)/2):floor((nx-1)/2))  * dk_x);
zz = single(reshape((-ceil((nz-1)/2):floor((nz-1)/2)) * dk_z, [1 1 nz]));

if useGPU
    yy = gpuArray(yy);
    xx = gpuArray(xx);
    zz = gpuArray(zz);
end

alpha = asin(min(single(na / nimm), single(0.999999)));
lambda_em = single(wavelength / nimm);
kz0 = single((1 - cos(alpha)) / lambda_em);
kz1 = single(sidebandFactor) * kz0;

nyq_lat = single(min(0.5 / dy, 0.5 / dx));
nyq_z   = single(0.5 / dz);
rdistcutoff = min(single(2 * na / wavelength), nyq_lat);
kz0 = min(kz0, nyq_z);
kz1 = min(kz1, nyq_z);

apocutoff = rdistcutoff + single(1 / linespacing);
zapocutoff = kz1;

rdistabs = sqrt(yy.^2 + xx.^2);   % [ny nx]
lat_norm2 = (rdistabs ./ max(apocutoff, eps('single'))).^2;
z_norm = abs(zz) ./ max(zapocutoff, eps('single'));   % [1 1 nz]

zmask0 = single(abs(zz) <= kz0);  % [1 1 nz]
zmask1 = single(abs(zz) <= kz1);  % [1 1 nz]

cudaFilter = struct();
cudaFilter.rdistcutoff = rdistcutoff;
cudaFilter.kz0 = kz0;
cudaFilter.kz1 = kz1;
cudaFilter.apocutoff = apocutoff;
cudaFilter.zapocutoff = zapocutoff;
cudaFilter.lat_norm2 = lat_norm2;
cudaFilter.z_norm = z_norm;
cudaFilter.apodizeoutput = apodizeoutput;
cudaFilter.zmask0 = zmask0;
cudaFilter.zmask1 = zmask1;
cudaFilter.zmask0_4d = reshape(zmask0, [1 1 nz 1]);
cudaFilter.zmask1_4d = reshape(zmask1, [1 1 nz 1]);
end

function ft_apo = apply_cuda_style_apodization_3d_gpu(recon_ft, cudaFilter)
rho = sqrt(cudaFilter.lat_norm2 + cudaFilter.z_norm.^2);
rho_clipped = min(single(1), rho);

switch cudaFilter.apodizeoutput
    case 1
        apo_abs = cos(single(pi/2) * rho_clipped);
    case 2
        apo_abs = single(1) - rho_clipped;
    otherwise
        apo_abs = single(1);
end

if ~isscalar(apo_abs)
    apo_abs(rho > 1) = 0;
end

if isscalar(apo_abs)
    ft_apo = recon_ft;
else
    if ndims(recon_ft) == 3
        ft_apo = recon_ft .* apo_abs;
    else
        ft_apo = recon_ft .* reshape(apo_abs, size(apo_abs,1), size(apo_abs,2), size(apo_abs,3), 1);
    end
end
end


function x = maybe_gather(x)
if isa(x, 'gpuArray')
    x = gather(x);
end
end



function tf = otf_is_polar_struct(otf_in)
tf = isstruct(otf_in) && isfield(otf_in,'kind') && contains(lower(string(otf_in.kind)), 'polar_otf');
end

function n = otf_num_dirs(otf_in)
if otf_is_polar_struct(otf_in)
    n = double(otf_in.ndirs);
else
    sz = size(otf_in);
    if numel(sz) < 5
        n = 1;
    else
        n = sz(5);
    end
end
end

function OTF_out = resample_polar_otf_to_cartesian(otf_in, data_dims, pxl_data, norders, ndirs, otfRA)
ny = data_dims(1); nx = data_dims(2); nz = data_dims(3);
kr_axis = single(otf_in.kr_axis(:));
kz_axis = single(otf_in.kz_axis(:));
radial = otf_in.radial_profiles;
ndirs_otf = size(radial,4);

dk_data = 1 ./ ([ny, nx, nz] .* pxl_data);
map_yy = single((-ceil((ny-1)/2):floor((ny-1)/2)) * dk_data(1));
map_xx = single((-ceil((nx-1)/2):floor((nx-1)/2)) * dk_data(2));
map_zz = single((-ceil((nz-1)/2):floor((nz-1)/2)) * dk_data(3));
[FY2, FX2] = ndgrid(map_yy, map_xx);
RXY = sqrt(FY2.^2 + FX2.^2);

n_dirs_out = 1; if ~otfRA, n_dirs_out = ndirs; end
OTF_out = complex(zeros(ny, nx, nz, norders, n_dirs_out, 'single'));
method = 'linear';
for d = 1:n_dirs_out
    didx = min(d, ndirs_otf);
    for ord = 1:norders
        profile = radial(:,:,ord,didx); % [nKr, nKz]
        vol = complex(zeros(ny, nx, nz, 'single'));
        for iz = 1:nz
            kzq = map_zz(iz);
            re_kr = interp1(double(kz_axis), double(real(profile)).', double(kzq), method, 0).';
            im_kr = interp1(double(kz_axis), double(imag(profile)).', double(kzq), method, 0).';
            re_xy = interp1(double(kr_axis), double(re_kr), double(RXY), method, 0);
            im_xy = interp1(double(kr_axis), double(im_kr), double(RXY), method, 0);
            vol(:,:,iz) = complex(single(re_xy), single(im_xy));
        end
        OTF_out(:,:,:,ord,d) = vol;
    end
end
end