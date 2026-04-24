function [otf_data, meta] = make_otf_2beam_3d_canonical(psf_data, params)
% MAKE_OTF_2BEAM_3D  Generate Cartesian 3D OTFs for the 2-beam 3D SIM recon script.
%
% This function produces OTF volumes in the exact layout expected by
% sim_recon_2beam_3d:
%   otf_data(y, x, z, order, dir)
% where
%   order = 1 -> order-0 / widefield OTF
%   order = 2 -> order+1 / first-order OTF
% and all OTF volumes are FFT-shifted so DC is at the array center.
%
% INPUT
%   psf_data : one of
%       [ny, nx, nz, 3]          single-direction PSF, 3 phases
%       [ny, nx, nz, 3, ndirs]   multi-direction PSF, 3 phases per direction
%
%   params fields
%       Required:
%         .pxl_dim_psf   [dy dx dz] PSF voxel size in microns
%         .linespacing   SIM line spacing in microns
%         .k0angles      [1 x ndirs] pattern angles in radians
%       Optional:
%         .nphases           default 3
%         .ndirs             inferred from psf_data if omitted
%         .background        scalar background to subtract from every phase
%         .auto_background   default true; estimate per-phase border mean
%         .border_size       default 8; border width for background estimate
%         .napodize          default 10; cosine edge taper width in pixels (XY and Z)
%         .use_centering     default true
%         .phase_step        nm; if empty, assume equally spaced phases
%         .bead_diameter     default 0.12 microns; set <= 0 to disable
%         .min_bead_tf       default 0.02; floor for bead TF compensation
%         .cleanup_threshold default 0; relative magnitude threshold for zeroing
%         .tile_to_ndirs     default false; replicate single-dir OTF to ndirs
%         .output_file       optional .mat filename to save results
%         .wavelength       emission wavelength (microns or nm if >10)
%         .na               numerical aperture for support cleanup
%         .nimm             immersion refractive index for support cleanup
%         .cleanup_otf      default true if wavelength/na/nimm provided, else false
%         .cylindrical_average default true; radially average OTF like cudasirecon
%         .fixorigin        default [2 9]; extrapolate kr=0 from this radial range
%         .krmax            default 0; override radial support cutoff in kr pixels
%         .leavekz          default [0 0 0]; keep this many kz pixels around origin
%         .nocompen         default false; disable bead-size compensation
%
% OUTPUT
%   otf_data : [ny, nx, nz, 2, ndirs_out], single, complex, FFT-shifted
%   meta     : struct with centers, shifts, backgrounds, scaling, diagnostics

%% ------------------------- parameter parsing -------------------------
if nargin < 2
    error('Usage: [otf_data, meta] = make_otf_2beam_3d(psf_data, params)');
end

assert(ndims(psf_data) == 4 || ndims(psf_data) == 5, ...
    'psf_data must be [ny nx nz 3] or [ny nx nz 3 ndirs].');

if ndims(psf_data) == 4
    psf_data = reshape(psf_data, size(psf_data,1), size(psf_data,2), ...
        size(psf_data,3), size(psf_data,4), 1);
end

[ny, nx, nz, nphases_in, ndirs_in] = size(psf_data);

if ~isfield(params, 'nphases'),           params.nphases = 3;            end
if ~isfield(params, 'ndirs'),             params.ndirs = ndirs_in;       end
if ~isfield(params, 'background'),        params.background = [];        end
if ~isfield(params, 'auto_background'),   params.auto_background = true; end
if ~isfield(params, 'border_size'),       params.border_size = 8;        end
if ~isfield(params, 'napodize'),          params.napodize = 10;          end
if ~isfield(params, 'use_centering'),     params.use_centering = true;   end
if ~isfield(params, 'phase_step'),        params.phase_step = [];        end
if ~isfield(params, 'bead_diameter'),     params.bead_diameter = 0.12;   end
if ~isfield(params, 'min_bead_tf'),       params.min_bead_tf = 0.02;     end
if ~isfield(params, 'cleanup_threshold'), params.cleanup_threshold = 0;  end
if ~isfield(params, 'tile_to_ndirs'),     params.tile_to_ndirs = false;  end
if ~isfield(params, 'output_file'),       params.output_file = '';       end
if ~isfield(params, 'truncate_bead_tf_after_first_zero'), params.truncate_bead_tf_after_first_zero = false; end
if ~isfield(params, 'wavelength'),        params.wavelength = [];        end
if ~isfield(params, 'na'),                params.na = [];                end
if ~isfield(params, 'nimm'),              params.nimm = [];              end
if ~isfield(params, 'cleanup_otf'),       params.cleanup_otf = [];       end
if ~isfield(params, 'cylindrical_average'), params.cylindrical_average = true; end
if ~isfield(params, 'fixorigin'),         params.fixorigin = [2 9];      end
if ~isfield(params, 'krmax'),             params.krmax = 0;              end
if ~isfield(params, 'leavekz'),           params.leavekz = [0 0 0];     end
if ~isfield(params, 'nocompen'),          params.nocompen = false;       end
if ~isfield(params, 'cleanup_interp_method'), params.cleanup_interp_method = 'linear'; end
if ~isfield(params, 'return_cartesian_preview'), params.return_cartesian_preview = false; end
if ~isfield(params, 'force_polar_output'), params.force_polar_output = true; end

assert(params.nphases == 3, 'This function supports 2-beam, 3-phase data only.');
assert(nphases_in == 3, 'The 4th dimension of psf_data must have 3 phases.');
assert(isfield(params, 'pxl_dim_psf'), 'Missing params.pxl_dim_psf');
assert(isfield(params, 'linespacing'), 'Missing params.linespacing');
assert(isfield(params, 'k0angles'),    'Missing params.k0angles');
assert(numel(params.pxl_dim_psf) == 3, 'params.pxl_dim_psf must be [dy dx dz].');
assert(numel(params.k0angles) >= ndirs_in, ...
    'params.k0angles must contain at least one angle per input direction.');

psf_data = single(psf_data);

% Normalize wavelength to microns if provided in nm.
if ~isempty(params.wavelength)
    if params.wavelength > 10
        params.wavelength = params.wavelength * 1e-3;
    end
    params.wavelength = single(params.wavelength);
end
if isempty(params.cleanup_otf)
    params.cleanup_otf = ~isempty(params.wavelength) && ~isempty(params.na) && ~isempty(params.nimm);
end
params.cleanup_otf = logical(params.cleanup_otf);

%% ---------------------- output direction handling --------------------
ndirs_out = ndirs_in;
if params.tile_to_ndirs
    assert(ndirs_in == 1, 'tile_to_ndirs is only valid when PSF has one direction.');
    assert(params.ndirs >= 1, 'params.ndirs must be >= 1 when tile_to_ndirs is true.');
    ndirs_out = params.ndirs;
end

otf_cart_data = complex(zeros(ny, nx, nz, 2, ndirs_out, 'single'));
radial_data = [];

meta = struct();
meta.centers_subpix = zeros(ndirs_in, 3, 'single');
meta.center_shifts  = zeros(ndirs_in, 3, 'single');
meta.backgrounds    = zeros(ndirs_in, 3, 'single');
meta.scale_factors  = zeros(ndirs_in, 1, 'single');
meta.k0_phys        = zeros(ndirs_in, 3, 'single');
meta.order_names    = {'order0_widefield', 'order1_plus'};
meta.otf_format     = '[ny, nx, nz, 2, ndirs], centered DC, order0/order+1';
meta.cleanup = struct('cylindrical_average', logical(params.cylindrical_average), ...
    'cleanup_otf', logical(params.cleanup_otf), ...
    'fixorigin', params.fixorigin, ...
    'krmax', params.krmax, ...
    'leavekz', params.leavekz, ...
    'nocompen', logical(params.nocompen));
meta.kr_axis = [];
meta.kz_axis = [];
meta.radial_profiles = cell(ndirs_in, 2);

%% -------------------------- reusable masks ---------------------------
xy_mask = make_xy_apod_mask(ny, nx, params.napodize);
xy_mask = reshape(xy_mask, ny, nx, 1);
z_mask  = make_z_apod_mask(nz, params.napodize);

%% -------------------- separation matrix ------------------------------
if ~isempty(params.phase_step)
    sep = make_separation_matrix_2beam(params.nphases, params.phase_step, params.linespacing);
else
    sep = make_separation_matrix_2beam_equal(params.nphases);
end

sep_cond = cond(double(sep));
if sep_cond > 1e6
    warning('Separation matrix ill-conditioned (cond = %.3g). Using pinv.', sep_cond);
    sep_inv = single(pinv(double(sep)));
else
    sep_inv = single(double(sep) \ eye(size(sep)));
end
fprintf('Separation matrix condition number: %.4f\n', sep_cond);

%% -------------------- frequency grids --------------------------------
fy = fftshift_frequency_axis(ny, params.pxl_dim_psf(1));
fx = fftshift_frequency_axis(nx, params.pxl_dim_psf(2));
fz = fftshift_frequency_axis(nz, params.pxl_dim_psf(3));
[FY, FX, FZ] = ndgrid(single(fy), single(fx), single(fz));
RXY = sqrt(FY(:,:,1).^2 + FX(:,:,1).^2);
KMAG = sqrt(FY.^2 + FX.^2 + FZ.^2);
dkr = min(abs(single(fy(min(end,2)) - fy(1))), abs(single(fx(min(end,2)) - fx(1))));
if dkr <= 0
    dkr = single(1 / max(ny, nx));
end
kr_axis = single(0:dkr:max(RXY(:)));
meta.kr_axis = kr_axis(:).';
meta.kz_axis = single(fz(:));
radial_data = complex(zeros(numel(kr_axis), numel(fz), 2, ndirs_out, 'single'));

center_idx = [floor(ny/2)+1, floor(nx/2)+1, floor(nz/2)+1];

%% -------------------- bead TF (same for all orders) ------------------
% The bead is a real-space object; its Fourier transform H(|k|) is the same
% regardless of which illumination order we are looking at. The separated
% order-m band in real space is band_m(r) = PSF(r) * bead(r) * pattern_m(r),
% so in Fourier space each band is convolved with the same bead transfer
% function. This matches the current reconstruction pipeline, which stores
% centered Cartesian OTFs and shifts the sideband OTF later during recon.
if params.bead_diameter > 0 && ~params.nocompen
    bead_radius = params.bead_diameter / 2;
    H_bead = sphere_transfer_function(KMAG, bead_radius);
    if params.truncate_bead_tf_after_first_zero
        H_bead = truncate_bead_tf_after_first_zero_local(H_bead, KMAG, bead_radius, params.min_bead_tf);
    else
        H_bead = max(abs(H_bead), single(params.min_bead_tf)) .* sign_nonzero(H_bead);
    end
else
    H_bead = [];
end

%% ======================== main loop ==================================
for idir = 1:ndirs_in
    fprintf('\n--- OTF direction %d/%d (angle = %.4f rad) ---\n', ...
        idir, ndirs_in, params.k0angles(idir));

    psf_dir = single(psf_data(:,:,:,:,idir));

    %% --- background subtraction per phase ---
    for iph = 1:3
        if ~isempty(params.background)
            bkg = single(params.background);
        elseif params.auto_background
            bkg = estimate_background_3d(psf_dir(:,:,:,iph), params.border_size);
        else
            bkg = single(0);
        end
        meta.backgrounds(idir, iph) = bkg;
        psf_dir(:,:,:,iph) = psf_dir(:,:,:,iph) - bkg;
        psf_dir(psf_dir<0) = 0;
        psf_dir = psf_dir./sum(psf_dir(:));
    end
    fprintf('  Backgrounds: [%.1f, %.1f, %.1f]\n', meta.backgrounds(idir,:));

    %% --- XY and Z edge apodization ---
    % Suppresses spectral leakage from non-periodic boundaries in the PSF stack.
    if params.napodize > 0
        psf_dir = psf_dir .* xy_mask .* z_mask;
    end

    %% --- estimate and apply centering ---
    avg_psf = mean(psf_dir, 4);
    ctr = estimate_peak_3d_subpixel(avg_psf);
    meta.centers_subpix(idir, :) = ctr;
    fprintf('  Bead center: [%.2f, %.2f, %.2f]\n', ctr);

    if params.use_centering
        shift_pix = single(center_idx - ctr);
        for iph = 1:3
            psf_dir(:,:,:,iph) = shift_realspace_fourier(psf_dir(:,:,:,iph), shift_pix);
        end
        fprintf('  Applied shift: [%.2f, %.2f, %.2f] pixels\n', shift_pix);
    else
        shift_pix = single([0 0 0]);
    end
    meta.center_shifts(idir, :) = shift_pix;

    %% --- phase separation: [-1, 0, +1] ---
    P = reshape(psf_dir, [], 3);
    B = P * sep_inv.';
    bands = reshape(B, ny, nx, nz, 3);
    % bands(:,:,:,1) = order -1
    % bands(:,:,:,2) = order  0  (widefield)
    % bands(:,:,:,3) = order +1

    %% --- compute centered Cartesian OTFs ---
    % Order-0 is already centered at DC after separation.
    order0 = fftshift(fftn(ifftshift(bands(:,:,:,2))));

    % The +1 band is NOT centered at DC after separation. For a polar/radial
    % OTF representation, we must demodulate it back to the origin before
    % radial averaging; otherwise the off-axis lobe collapses toward zero.
    k0_phys = nominal_k0_phys(params.k0angles(idir), params.linespacing);
    [order1, demodInfo] = center_first_order_otf_from_band(bands(:,:,:,3), k0_phys, params.pxl_dim_psf);
    meta.demod_info(idir) = demodInfo;

    %% --- bead-size compensation ---
    if ~isempty(H_bead)
        order0 = order0 ./ H_bead;
        order1 = order1 ./ H_bead;
    end

    %% --- cudasirecon-style radial averaging / origin fix / support cleanup ---
    if params.cylindrical_average || params.cleanup_otf || any(params.krmax ~= 0)
        [order0, rad0, cleanup0] = cudasirecon_style_otf_cleanup(order0, RXY, fz, kr_axis, params, 0);
        [order1, rad1, cleanup1] = cudasirecon_style_otf_cleanup(order1, RXY, fz, kr_axis, params, 1);
        meta.radial_profiles{idir,1} = rad0;
        meta.radial_profiles{idir,2} = rad1;
        meta.cleanup_info(idir).order0 = cleanup0;
        meta.cleanup_info(idir).order1 = cleanup1;
    else
        meta.radial_profiles{idir,1} = [];
        meta.radial_profiles{idir,2} = [];
        meta.cleanup_info(idir).order0 = struct();
        meta.cleanup_info(idir).order1 = struct();
    end

    meta.k0_phys(idir, :) = nominal_k0_phys(params.k0angles(idir), params.linespacing);

    %% --- cleanup tiny values ---
    if params.cleanup_threshold > 0
        thr0 = single(params.cleanup_threshold) * max(abs(order0(:)));
        thr1 = single(params.cleanup_threshold) * max(abs(order1(:)));
        order0(abs(order0) < thr0) = 0;
        order1(abs(order1) < thr1) = 0;
    end

    %% --- normalize by order-0 DC ---
    dc0 = order0(center_idx(1), center_idx(2), center_idx(3));
    if abs(dc0) < eps('single')
        warning('Direction %d: order-0 DC is near zero; skipping normalization.', idir);
        scale = single(1);
    else
        scale = single(1 / dc0);
    end
    meta.scale_factors(idir) = scale;

    order0 = single(order0 * scale);
    order1 = single(order1 * scale);

    fprintf('  DC scale factor: %.6f\n', abs(scale));
    fprintf('  Order-0 peak: %.6f, Order-1 peak: %.6f\n', max(abs(order0(:))), max(abs(order1(:))));

    otf_cart_data(:,:,:,1,idir) = order0;
    otf_cart_data(:,:,:,2,idir) = order1;
    if ~isempty(meta.radial_profiles{idir,1}), radial_data(:,:,1,idir) = meta.radial_profiles{idir,1}; else, radial_data(:,:,1,idir) = radial_average_otf_cartesian(order0, RXY, kr_axis); end
    if ~isempty(meta.radial_profiles{idir,2}), radial_data(:,:,2,idir) = meta.radial_profiles{idir,2}; else, radial_data(:,:,2,idir) = radial_average_otf_cartesian(order1, RXY, kr_axis); end
end

%% -------------------------- tile if requested ------------------------
if params.tile_to_ndirs && ndirs_out > 1
    otf_cart_data = repmat(otf_cart_data(:,:,:,:,1), 1, 1, 1, 1, ndirs_out);
    radial_data = repmat(radial_data(:,:,:,1), 1, 1, 1, ndirs_out);
    fprintf('\nTiled single-direction OTF to %d directions.\n', ndirs_out);
end


%% -------------------------- package output -------------------------
if params.force_polar_output
    otf_data = struct();
    otf_data.kind = 'polar_otf_2beam3d';
    otf_data.kr_axis = single(kr_axis(:).');
    otf_data.kz_axis = single(fz(:).');
    otf_data.radial_profiles = radial_data;    % [nKr nz 2 ndirs]
    otf_data.ndirs = ndirs_out;
    otf_data.norders = 2;
    otf_data.otfRA = (ndirs_out == 1);
    otf_data.pxl_dim_psf = single(params.pxl_dim_psf(:).');
    otf_data.linespacing = single(params.linespacing);
    otf_data.k0angles = single(params.k0angles(1:ndirs_out));
    otf_data.cleanup = meta.cleanup;
    otf_data.cleanup_info = meta.cleanup_info;
    otf_data.cartesian_reference_size = int32([ny nx nz]);
    otf_data.return_cartesian_preview = logical(params.return_cartesian_preview);
    if params.return_cartesian_preview
        otf_data.cartesian_preview = otf_cart_data;
    else
        otf_data.cartesian_preview = [];
    end
    meta.output_kind = otf_data.kind;
    meta.radial_size = int32(size(radial_data));
else
    otf_data = otf_cart_data;
    meta.output_kind = 'cartesian_otf_from_radial_cleanup';
    meta.radial_size = int32(size(radial_data));
    meta.cartesian_preview = [];
end

%% -------------------------- optional save ----------------------------
if ~isempty(params.output_file)
    save(params.output_file, 'otf_data', 'meta', '-v7.3');
    fprintf('\nOTF saved to %s\n', params.output_file);
end

if params.force_polar_output
    fprintf('\nOTF generation complete (polar): [nKr=%d, nz=%d, 2, ndirs=%d]\n', ...
        numel(kr_axis), numel(fz), ndirs_out);
else
    fprintf('\nOTF generation complete (cartesian): [%d, %d, %d, 2, %d]\n', ny, nx, nz, ndirs_out);
end
end

%% ====================================================================
%  HELPER FUNCTIONS
%  ====================================================================

function sep = make_separation_matrix_2beam_equal(nphases)
orders = [-1, 0, 1];
sep = complex(zeros(nphases, numel(orders), 'single'));
for j = 1:nphases
    phi = single((j-1) * 2*pi / nphases);
    for m = 1:numel(orders)
        sep(j,m) = exp(1i * single(orders(m)) * phi);
    end
end
end

function sep = make_separation_matrix_2beam(nphases, phase_step_nm, linespacing_um)
delta_phi = single((phase_step_nm * 1e-3) / linespacing_um * 2*pi);
orders = [-1, 0, 1];
sep = complex(zeros(nphases, numel(orders), 'single'));
for j = 1:nphases
    for m = 1:numel(orders)
        sep(j,m) = exp(1i * single(orders(m)) * single(j-1) * delta_phi);
    end
end
end

function mask = make_xy_apod_mask(ny, nx, napodize)
% Cosine-taper mask for XY edges.
if napodize <= 0
    mask = ones(ny, nx, 'single');
    return;
end
napodize = min([napodize, floor(ny/2), floor(nx/2)]);
y = ones(ny, 1, 'single');
x = ones(1, nx, 'single');
if napodize > 0
    ramp = single(0.5 * (1 - cos(pi * (0:napodize-1).' / napodize)));
    y(1:napodize)           = ramp;
    y(end-napodize+1:end)   = flipud(ramp);
    x(1,1:napodize)         = ramp.';
    x(1,end-napodize+1:end) = fliplr(ramp.');
end
mask = y .* x;
end

function mask = make_z_apod_mask(nz, napodize)
% Cosine-taper mask for Z edges.
mask = ones(1, 1, nz, 'single');
nap = min(napodize, floor(nz/2));
if nap > 0
    ramp = single(0.5 * (1 - cos(pi * (0:nap-1).' / nap)));
    mask(1,1,1:nap)       = reshape(ramp, 1, 1, []);
    mask(1,1,nz-nap+1:nz) = reshape(flipud(ramp), 1, 1, []);
end
end

function bkg = estimate_background_3d(vol, border_size)
% Estimate background from border voxels of a 3D volume.
[ny, nx, nz] = size(vol);
b = min([border_size, floor(ny/2)-1, floor(nx/2)-1, max(1, floor(nz/2)-1)]);
if b <= 0
    bkg = single(mean(vol(:), 'omitnan'));
    return;
end
mask = false(ny, nx, nz);
mask(1:b,:,:)               = true;
mask(end-b+1:end,:,:)       = true;
mask(:,1:b,:)               = true;
mask(:,end-b+1:end,:)       = true;
mask(:,:,1:min(b,nz))       = true;
mask(:,:,max(1,nz-b+1):nz)  = true;
vals = vol(mask);
if isempty(vals)
    bkg = single(mean(vol(:), 'omitnan'));
else
    bkg = single(mean(vals, 'omitnan'));
end
end

function ctr = estimate_peak_3d_subpixel(vol)
% Find peak location with sub-pixel refinement via parabolic interpolation.
[ny, nx, nz] = size(vol);
[~, idx] = max(vol(:));
[py, px, pz] = ind2sub(size(vol), idx);

ctr = single([ ...
    subpixel_peak_1d(vol(:, px, pz), py), ...
    subpixel_peak_1d(squeeze(vol(py, :, pz)).', px), ...
    subpixel_peak_1d(squeeze(vol(py, px, :)), pz)]);

ctr(1) = min(max(ctr(1), 1), ny);
ctr(2) = min(max(ctr(2), 1), nx);
ctr(3) = min(max(ctr(3), 1), nz);
end

function pos = subpixel_peak_1d(v, idx)
% 1D parabolic sub-pixel peak refinement.
v = single(v(:));
if idx <= 1 || idx >= numel(v)
    pos = single(idx);
    return;
end
a1 = v(idx-1);
a2 = v(idx);
a3 = v(idx+1);
slope = 0.5 * (a3 - a1);
curve = (a3 + a1) - 2*a2;
if abs(curve) < eps('single')
    delta = single(0);
else
    delta = -slope / curve;
    delta = max(min(delta, 1.5), -1.5);
end
pos = single(idx) + delta;
end

function vol_shifted = shift_realspace_fourier(vol, shift_pix)
[ny, nx, nz] = size(vol);
F = fftn(vol);
ky = ifftshift(-ceil((ny-1)/2):floor((ny-1)/2)) / ny;
kx = ifftshift(-ceil((nx-1)/2):floor((nx-1)/2)) / nx;
kz = ifftshift(-ceil((nz-1)/2):floor((nz-1)/2)) / nz;
py = exp(-1i * 2*pi * single(shift_pix(1)) * reshape(single(ky), [ny 1 1]));
px = exp(-1i * 2*pi * single(shift_pix(2)) * reshape(single(kx), [1 nx 1]));
pz = exp(-1i * 2*pi * single(shift_pix(3)) * reshape(single(kz), [1 1 nz]));
vol_shifted = real(ifftn(F .* py .* px .* pz));
vol_shifted = single(vol_shifted);
end

function f = fftshift_frequency_axis(N, d)
% Frequency axis in cycles/micron, in fftshift order (DC at center).
idx = -ceil((N-1)/2):floor((N-1)/2);
f = idx / (N * d);
end

function k0 = nominal_k0_phys(angle_rad, linespacing)
% Pattern wave-vector in cycles/micron. k0z = 0 for 2-beam lateral SIM.
k0_mag = 1 / linespacing;
k0 = single([k0_mag * cos(angle_rad), k0_mag * sin(angle_rad), 0]);
end

function H = sphere_transfer_function(kmag, radius)
% Fourier transform of a uniform sphere (normalized to H(0)=1).
%   kmag   : spatial frequency magnitude in cycles/micron
%   radius : sphere radius in microns
% Returns the signed real transfer function:
%   H(x) = 3(sin(x) - x cos(x)) / x^3,  x = 2π·radius·|k|
x = 2*pi * radius * single(kmag);
H = ones(size(x), 'like', x);
mask = abs(x) > 1e-6;
xx = x(mask);
H(mask) = 3 * (sin(xx) - xx .* cos(xx)) ./ (xx.^3);
H = real(H);
end

function s = sign_nonzero(x)
% Like sign(x), but zeros map to +1 so floors preserve phase sign safely.
s = ones(size(x), 'like', x);
s(real(x) < 0) = -1;
end

function [order1_centered, info] = center_first_order_otf_from_band(band_p1_real, k0_phys, pxl_dim_psf)
% Demodulate the separated +1 band back to DC before FFT/radial averaging.
[ny, nx, nz] = size(band_p1_real);
cy = floor(ny/2);
cx = floor(nx/2);
yy = single((0:ny-1) - cy) * single(pxl_dim_psf(1));
xx = single((0:nx-1) - cx) * single(pxl_dim_psf(2));
[YY, XX] = ndgrid(yy, xx);
phase_arg = 2*pi * (single(k0_phys(1)) * YY + single(k0_phys(2)) * XX);
ramp_minus = exp(-1i * phase_arg);
ramp_plus  = exp( 1i * phase_arg);

cand_minus = fftshift(fftn(ifftshift(band_p1_real .* ramp_minus)));
cand_plus  = fftshift(fftn(ifftshift(band_p1_real .* ramp_plus)));

cidx = [floor(ny/2)+1, floor(nx/2)+1, floor(nz/2)+1];
score_minus = abs(cand_minus(cidx(1), cidx(2), cidx(3)));
score_plus  = abs(cand_plus(cidx(1), cidx(2), cidx(3)));

if score_minus >= score_plus
    order1_centered = cand_minus;
    chosen_sign = -1;
    chosen_score = score_minus;
else
    order1_centered = cand_plus;
    chosen_sign = +1;
    chosen_score = score_plus;
end

info = struct('chosen_sign', single(chosen_sign), ...
              'score_minus', single(score_minus), ...
              'score_plus', single(score_plus), ...
              'chosen_score', single(chosen_score), ...
              'k0_phys', single(k0_phys));
end


function [otf_cart, radial_profile, info] = cudasirecon_style_otf_cleanup(otf_cart_in, RXY, fz, kr_axis, params, order_idx)
% Best-effort MATLAB emulation of cudasirecon makeOTF processing while
% still returning a Cartesian OTF volume for the MATLAB recon path.
%
% Steps:
%   1) cylindrical/radial averaging in (kr, kz)
%   2) optional kr=0 origin fix by extrapolation over fixorigin range
%   3) optional support cleanup using wavelength/NA/nimm or user krmax
%   4) map the cleaned radial OTF back to Cartesian coordinates

[ny, nx, nz] = size(otf_cart_in);
nKr = numel(kr_axis);

radial_profile = radial_average_otf_cartesian(otf_cart_in, RXY, kr_axis);

if ~isempty(params.fixorigin)
    radial_profile = fixorigin_extrapolation(radial_profile, params.fixorigin, kr_axis);
end

info = struct();
info.order_index = order_idx;
info.used_cylindrical_average = logical(params.cylindrical_average);
info.used_cleanup = logical(params.cleanup_otf);
info.krmax_px = 0;
info.kzcut_um_inv = [];
info.cleanup_mask_fraction = 1;

if params.cleanup_otf || any(params.krmax ~= 0)
    [radial_profile, info] = cleanup_radial_otf_support(radial_profile, kr_axis, single(fz(:)), params, order_idx);
end

if params.cylindrical_average || params.cleanup_otf || any(params.krmax ~= 0)
    otf_cart = radial_otf_to_cartesian(radial_profile, RXY, kr_axis, params.cleanup_interp_method);
else
    otf_cart = otf_cart_in;
end
end

function radial_profile = radial_average_otf_cartesian(otf_cart, RXY, kr_axis)
[~, ~, nz] = size(otf_cart);
nKr = numel(kr_axis);
dkr = kr_axis(2) - kr_axis(1);
if dkr <= 0
    dkr = single(1);
end

idx = round(RXY / dkr) + 1;
idx = max(1, min(nKr, idx));
idxv = idx(:);

radial_profile = complex(zeros(nKr, nz, 'single'));

for iz = 1:nz
    sl = otf_cart(:,:,iz);
    re = accumarray(double(idxv), double(real(sl(:))), [nKr 1], @mean, NaN);
    im = accumarray(double(idxv), double(imag(sl(:))), [nKr 1], @mean, NaN);

    % Fill any NaNs by nearest valid samples to avoid holes.
    re = fillmissing_local(single(re));
    im = fillmissing_local(single(im));
    radial_profile(:, iz) = complex(re, im);
end
end

function v = fillmissing_local(v)
if all(~isfinite(v))
    v(:) = 0;
    return;
end
good = find(isfinite(v));
bad = find(~isfinite(v));
for k = 1:numel(bad)
    [~, ii] = min(abs(good - bad(k)));
    v(bad(k)) = v(good(ii));
end
end

function radial_profile = fixorigin_extrapolation(radial_profile, fixorigin, kr_axis)
nKr = size(radial_profile, 1);
if isempty(fixorigin)
    return;
end
if isscalar(fixorigin)
    fit_idx = [2, round(fixorigin)];
else
    fit_idx = round(fixorigin(:).');
end
fit_idx(1) = max(2, fit_idx(1));
fit_idx(2) = min(nKr, fit_idx(2));
if fit_idx(2) <= fit_idx(1)
    return;
end

x = double(kr_axis(fit_idx(1):fit_idx(2)));
for iz = 1:size(radial_profile, 2)
    yre = double(real(radial_profile(fit_idx(1):fit_idx(2), iz)));
    yim = double(imag(radial_profile(fit_idx(1):fit_idx(2), iz)));

    if any(isfinite(yre))
        pre = polyfit(x(:), yre(:), 1);
        radial_profile(1, iz) = complex(single(polyval(pre, 0)), imag(radial_profile(1, iz)));
    end
    if any(isfinite(yim))
        pim = polyfit(x(:), yim(:), 1);
        radial_profile(1, iz) = complex(real(radial_profile(1, iz)), single(polyval(pim, 0)));
    end
end
end

function [radial_profile, info] = cleanup_radial_otf_support(radial_profile, kr_axis, kz_axis, params, order_idx)
info = struct();
info.order_index = order_idx;
info.krmax_px = 0;
info.kzcut_um_inv = [];
info.cleanup_mask_fraction = 1;

nKr = size(radial_profile, 1);
nZ = size(radial_profile, 2);

% kr cutoff
if ~isempty(params.krmax) && any(params.krmax ~= 0)
    if numel(params.krmax) >= order_idx + 1
        krmax_px = round(params.krmax(min(numel(params.krmax), order_idx + 1)));
    else
        krmax_px = round(params.krmax(1));
    end
else
    krmax_px = nKr;
    if ~isempty(params.wavelength) && ~isempty(params.na)
        krmax_um_inv = single(2 * params.na / params.wavelength);
        krmax_px = min(nKr, floor(krmax_um_inv / max(kr_axis(2)-kr_axis(1), eps('single'))) + 1);
    end
end
krmax_px = max(1, min(nKr, krmax_px));

% exact axial cutoff at kr=0 from cudasirecon-style expression
kzcut = inf;
if ~isempty(params.wavelength) && ~isempty(params.na) && ~isempty(params.nimm)
    alpha = asin(min(single(params.na / params.nimm), single(1)));
    lambda_em = single(params.wavelength / params.nimm);
    kzcut = single((1 - cos(alpha)) / lambda_em);
end

leavekz = params.leavekz;
if isempty(leavekz)
    leavekz = [0 0 0];
end
if isscalar(leavekz)
    leavekz = [leavekz leavekz leavekz];
end
leavekz = abs(round(double(leavekz(:).')));
lk_idx = min(numel(leavekz), order_idx + 1);
leavekz_px = leavekz(lk_idx);

kz_center = floor(nZ/2) + 1;
mask_kr = (1:nKr) <= krmax_px;
mask_kz = true(1, nZ);
if isfinite(kzcut)
    mask_kz = abs(single(kz_axis(:)).') <= kzcut;
    if leavekz_px > 0
        kk = max(1, kz_center-leavekz_px):min(nZ, kz_center+leavekz_px);
        mask_kz(kk) = true;
    end
end

mask = mask_kr(:) * mask_kz(:).';
radial_profile(~mask) = 0;

info.krmax_px = krmax_px;
info.kzcut_um_inv = kzcut;
info.cleanup_mask_fraction = nnz(mask) / numel(mask);
end

function otf_cart = radial_otf_to_cartesian(radial_profile, RXY, kr_axis, interp_method)
[ny, nx] = size(RXY);
nz = size(radial_profile, 2);
otf_cart = complex(zeros(ny, nx, nz, 'single'));

rquery = double(RXY);
x = double(kr_axis(:));

for iz = 1:nz
    yre = double(real(radial_profile(:, iz)));
    yim = double(imag(radial_profile(:, iz)));

    re = interp1(x, yre, rquery, interp_method, 0);
    im = interp1(x, yim, rquery, interp_method, 0);
    otf_cart(:,:,iz) = complex(single(re), single(im));
end
end

function H = truncate_bead_tf_after_first_zero_local(H, kmag, radius, min_tf)
first_zero_x = single(4.493409457909064);
k_zero = first_zero_x / single(2*pi*radius);
mask = kmag >= k_zero;
H(mask) = single(min_tf);
H(~mask) = max(H(~mask), single(min_tf));
H = single(H);
end

