function fitReport = refit_sim_params_2beam_3d(raw_data, params)
% REFIT_SIM_PARAMS_2BEAM_3D
% Global, z-summed analytical refinement of carrier angle, line spacing,
% and phase vector for 2-beam, 3-phase SIM.
%
% Input raw_data may be either:
%   [ny, nx, nz*nphases*ndirs]   flattened Angle->Z->Phase order
%   [ny, nx, nz, nphases, ndirs] reshaped stack

if ~isfield(params,'nphases'), params.nphases = 3; end
if ~isfield(params,'ndirs'), error('params.ndirs required'); end
if ~isfield(params,'pxl_dim_data'), error('params.pxl_dim_data required'); end
if ~isfield(params,'k0angles'), error('params.k0angles required'); end
if ~isfield(params,'linespacing'), error('params.linespacing required'); end
if ~isfield(params,'background'), params.background = 0; end
if ~isfield(params,'fastSI'), params.fastSI = false; end
if ~isfield(params,'phase_step'), params.phase_step = []; end
if ~isfield(params,'phase_vector_rad'), params.phase_vector_rad = []; end
if ~isfield(params,'refit_k0_search_window_px'), params.refit_k0_search_window_px = 6; end
if ~isfield(params,'refit_apodize_xy_px'), params.refit_apodize_xy_px = 10; end
if ~isfield(params,'otf_support_thresh'), params.otf_support_thresh = 0.006; end

nphases = params.nphases;
ndirs = params.ndirs;
assert(nphases == 3, 'Only 3-phase data supported.');

if ndims(raw_data) == 3
    [ny,nx,nimgs] = size(raw_data);
    nz = nimgs / (nphases * ndirs);
    assert(mod(nz,1)==0, 'raw_data page count incompatible with nphases*ndirs');
    nz = round(nz);
    raw_data = single(raw_data) - single(params.background);
    if params.fastSI
        data_5d = reshape(raw_data, ny, nx, nphases, ndirs, nz);
        data_5d = permute(data_5d, [1 2 5 3 4]);
    else
        data_5d = reshape(raw_data, ny, nx, nphases, nz, ndirs);
        data_5d = permute(data_5d, [1 2 4 3 5]);
    end
elseif ndims(raw_data) == 5
    data_5d = single(raw_data);
    [ny,nx,nz,nphases_in,ndirs_in] = size(data_5d);
    assert(nphases_in == nphases && ndirs_in == ndirs, 'raw_data size mismatch');
    data_5d = data_5d - single(params.background);
else
    error('raw_data must be 3D flattened or 5D reshaped');
end

mask2d = make_xy_apod_mask_2d(ny, nx, params.refit_apodize_xy_px);
dk_y = 1 / (ny * params.pxl_dim_data(1));
dk_x = 1 / (nx * params.pxl_dim_data(2));

fitReport = struct();
fitReport.method = 'global z-summed analytical carrier/phase refinement';
fitReport.note = ['Carrier is refined from the Band0/Band+1 cross-correlation peak. ' ...
    'Phase vector is estimated from the complex raw FFT values sampled at the refined carrier.'];
fitReport.phase_vectors_rad = zeros(ndirs, 3, 'single');
fitReport.k0angles_refined = zeros(1, ndirs, 'single');
fitReport.linespacing_refined = zeros(1, ndirs, 'single');
fitReport.k0_pix_refined = zeros(ndirs, 3, 'single');
fitReport.background_residual = zeros(ndirs, 3, 'single');
fitReport.directions = repmat(struct(), 1, ndirs);

for idir = 1:ndirs
    phase_expected = resolve_phase_vector_for_dir(params, idir);

    proj = squeeze(sum(data_5d(:,:,:,:,idir), 3)); % [ny nx 3]
    for iph = 1:3
        proj(:,:,iph) = proj(:,:,iph) .* mask2d;
        fitReport.background_residual(idir, iph) = mean(proj([1 end],:,iph), 'all');
    end

    F = complex(zeros(ny, nx, 3, 'single'));
    for iph = 1:3
        F(:,:,iph) = fftshift2(fft2(proj(:,:,iph)));
    end

    sep0 = make_sep_from_phases(phase_expected);
    sep_inv0 = single(pinv(double(sep0)));
    bands0 = separate_2d(F, sep_inv0);

    k0_nom = compute_nominal_k0_2d(params.k0angles(idir), params.linespacing, dk_y, dk_x);
    k0_refined_xy = find_k0_from_bands2d(bands0(:,:,2), bands0(:,:,3), ...
        k0_nom, params.refit_k0_search_window_px);

    raw_samples = complex(zeros(1,3,'single'));
    for iph = 1:3
        raw_samples(iph) = sample_complex_bilinear_2d(F(:,:,iph), k0_refined_xy);
    end
    phase_fit = estimate_phase_vector_from_samples(raw_samples, phase_expected);

    sep1 = make_sep_from_phases(phase_fit);
    sep_inv1 = single(pinv(double(sep1)));
    bands1 = separate_2d(F, sep_inv1);
    k0_refined_xy = find_k0_from_bands2d(bands1(:,:,2), bands1(:,:,3), ...
        k0_refined_xy, max(2, round(params.refit_k0_search_window_px/2)));

    for iph = 1:3
        raw_samples(iph) = sample_complex_bilinear_2d(F(:,:,iph), k0_refined_xy);
    end
    phase_fit = estimate_phase_vector_from_samples(raw_samples, phase_expected);

    sep_final = make_sep_from_phases(phase_fit);
    sep_inv_final = single(pinv(double(sep_final)));
    bands_final = separate_2d(F, sep_inv_final);
    score = compute_overlap_score_2d(bands_final(:,:,2), bands_final(:,:,3), ...
        k0_refined_xy, params.otf_support_thresh);

    ky_phys = k0_refined_xy(1) * dk_y;
    kx_phys = k0_refined_xy(2) * dk_x;
    k_mag = sqrt(ky_phys.^2 + kx_phys.^2);
    ls_fit = 1 / max(k_mag, eps('single'));
    ang_fit = atan2(kx_phys, ky_phys);

    phase_deg_fit = rad2deg(double(phase_fit));
    phase_deg_exp = rad2deg(double(phase_expected));
    step_fit_deg = diff(unwrap(double(phase_fit))) * 180/pi;
    step_exp_deg = diff(unwrap(double(phase_expected))) * 180/pi;

    fitReport.phase_vectors_rad(idir,:) = single(phase_fit);
    fitReport.k0angles_refined(idir) = single(ang_fit);
    fitReport.linespacing_refined(idir) = single(ls_fit);
    fitReport.k0_pix_refined(idir,:) = single([k0_refined_xy, 0]);
    fitReport.directions(idir).sep_inv = single(sep_inv_final);
    fitReport.directions(idir).noise_weights = single(1 ./ max(sum(abs(sep_inv_final).^2,2).', eps('single')));
    fitReport.directions(idir).score = single(score);
    fitReport.directions(idir).raw_samples = raw_samples;
    fitReport.directions(idir).phase_deg_fit = single(phase_deg_fit);
    fitReport.directions(idir).phase_deg_expected = single(phase_deg_exp);
    fitReport.directions(idir).phase_deg_delta = single(phase_deg_fit - phase_deg_exp);
    fitReport.directions(idir).step_deg_fit = single(step_fit_deg);
    fitReport.directions(idir).step_deg_expected = single(step_exp_deg);
    fitReport.directions(idir).step_pct_delta = single(100 * (step_fit_deg - step_exp_deg) ./ max(abs(step_exp_deg), eps));
    fitReport.directions(idir).linespacing_pct_delta = single(100 * (ls_fit - params.linespacing) / params.linespacing);
    fitReport.directions(idir).angle_deg_delta = single(rad2deg(wrap_to_pi_local(ang_fit - params.k0angles(idir))));

    fprintf('\nRefit dir %d/%d\n', idir, ndirs);
    fprintf('  line spacing: fitted %.6f um vs input %.6f um (%+0.2f%%)\n', ...
        ls_fit, params.linespacing, fitReport.directions(idir).linespacing_pct_delta);
    fprintf('  angle:        fitted %.3f deg vs input %.3f deg (%+0.3f deg)\n', ...
        rad2deg(ang_fit), rad2deg(params.k0angles(idir)), fitReport.directions(idir).angle_deg_delta);
    fprintf('  phases:       fitted [%.2f %.2f %.2f] deg vs expected [%.2f %.2f %.2f] deg\n', ...
        phase_deg_fit(1), phase_deg_fit(2), phase_deg_fit(3), ...
        phase_deg_exp(1), phase_deg_exp(2), phase_deg_exp(3));
    fprintf('  step error:   [%+.2f%%, %+.2f%%], score %.4f\n', ...
        fitReport.directions(idir).step_pct_delta(1), fitReport.directions(idir).step_pct_delta(2), score);
end

fitReport.linespacing_refined_global = single(mean(fitReport.linespacing_refined));
end

function phase_expected = resolve_phase_vector_for_dir(params, idir)
if ~isempty(params.phase_vector_rad)
    pv = params.phase_vector_rad;
    if isvector(pv)
        phase_expected = single(reshape(pv, 1, 3));
    else
        phase_expected = single(pv(idir,:));
    end
elseif ~isempty(params.phase_step)
    delta_exp = (params.phase_step * 1e-3 / params.linespacing) * 2*pi;
    phase_expected = single([0, delta_exp, 2*delta_exp]);
else
    phase_expected = single([0, 2*pi/3, 4*pi/3]);
end
phase_expected = align_phase_vector_monotonic(phase_expected);
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

function bands = separate_2d(F, sep_inv)
[ny, nx, nph] = size(F);
P = reshape(F, [], nph);
B = P * sep_inv.';
bands = reshape(B, ny, nx, nph);
end

function k0_nom = compute_nominal_k0_2d(angle_rad, linespacing, dk_y, dk_x)
k0_phys = 1 / linespacing;
k0_nom = single([k0_phys * cos(angle_rad) / dk_y, k0_phys * sin(angle_rad) / dk_x]);
end

function k0_pix = find_k0_from_bands2d(band0, band1, k0_nominal_px, win)
xcorr_xy = fftshift2(ifft2(ifftshift2(conj(band0) .* band1)));
[ny, nx] = size(xcorr_xy);
cy = floor(ny/2) + 1;
cx = floor(nx/2) + 1;
pk_y = round(k0_nominal_px(1)) + cy;
pk_x = round(k0_nominal_px(2)) + cx;
y_range = max(1, pk_y-win):min(ny, pk_y+win);
x_range = max(1, pk_x-win):min(nx, pk_x+win);
patch = abs(xcorr_xy(y_range, x_range));
[~, idx] = max(patch(:));
[iy, ix] = ind2sub(size(patch), idx);
[dy, dx] = subpixel_refine_patch_2d(patch, iy, ix);
k0_pix = single([y_range(iy) + dy - cy, x_range(ix) + dx - cx]);
end

function c = sample_complex_bilinear_2d(F, k0_pix)
[ny, nx] = size(F);
cy = floor(ny/2) + 1;
cx = floor(nx/2) + 1;
yy = cy + double(k0_pix(1));
xx = cx + double(k0_pix(2));

y0 = floor(yy); x0 = floor(xx);
ay = yy - y0;   ax = xx - x0;

y0 = max(1, min(ny-1, y0));
x0 = max(1, min(nx-1, x0));
y1 = y0 + 1;
x1 = x0 + 1;

c00 = F(y0, x0);
c01 = F(y0, x1);
c10 = F(y1, x0);
c11 = F(y1, x1);

c = single((1-ay)*(1-ax)*c00 + (1-ay)*ax*c01 + ay*(1-ax)*c10 + ay*ax*c11);
end

function phase_fit = estimate_phase_vector_from_samples(raw_samples, phase_expected)
phi_raw = unwrap(double(angle(raw_samples)));
phi_rel = phi_raw - phi_raw(1);
phase_fit = zeros(1,3,'single');
phase_fit(1) = 0;
for j = 2:3
    candidates = phi_rel(j) + 2*pi*(-2:2);
    [~,idx] = min(abs(candidates - double(phase_expected(j))));
    phase_fit(j) = single(candidates(idx));
end
phase_fit = align_phase_vector_monotonic(phase_fit);
end

function score = compute_overlap_score_2d(band0, band1, k0_pix, supportThresh)
shifted = shift2d_fourier(band1, k0_pix);
a0 = abs(band0);
a1 = abs(shifted);
mask = (a0 > supportThresh * max(a0(:))) & (a1 > supportThresh * max(a1(:)));
if nnz(mask) < 10
    score = -inf;
    return;
end
num = abs(sum(conj(band0(mask)) .* shifted(mask)));
den = sqrt(sum(abs(band0(mask)).^2) * sum(abs(shifted(mask)).^2) + eps('single'));
score = single(num / den);
end

function shifted = shift2d_fourier(img, shift_pix)
[ny, nx] = size(img);
int_shift = round(shift_pix);
frac_shift = shift_pix - int_shift;
shifted = circshift(img, int_shift);
if any(abs(frac_shift) > 1e-6)
    tmp = ifft2(ifftshift2(shifted));
    ky = ifftshift(-ceil((ny-1)/2):floor((ny-1)/2)) / ny;
    kx = ifftshift(-ceil((nx-1)/2):floor((nx-1)/2)) / nx;
    py = exp(1i * single(2*pi*frac_shift(1)) * reshape(single(ky), [ny 1]));
    px = exp(1i * single(2*pi*frac_shift(2)) * reshape(single(kx), [1 nx]));
    shifted = fftshift2(fft2(tmp .* py .* px));
end
end

function out = fftshift2(in)
out = fftshift(in,1); out = fftshift(out,2);
end

function out = ifftshift2(in)
out = ifftshift(in,1); out = ifftshift(out,2);
end

function mask = make_xy_apod_mask_2d(ny, nx, napodize)
if napodize <= 0
    mask = ones(ny, nx, 'single');
    return;
end
napodize = min([napodize, floor(ny/2), floor(nx/2)]);
y = ones(ny, 1, 'single');
x = ones(1, nx, 'single');
if napodize > 0
    ramp = single(0.5 * (1 - cos(pi * (0:napodize-1).' / napodize)));
    y(1:napodize) = ramp;
    y(end-napodize+1:end) = flipud(ramp);
    x(1,1:napodize) = ramp.';
    x(1,end-napodize+1:end) = fliplr(ramp.');
end
mask = y .* x;
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

function phase_vec = align_phase_vector_monotonic(phase_vec)
phase_vec = single(phase_vec(:).');
phase_vec = phase_vec - phase_vec(1);
for j = 2:numel(phase_vec)
    while phase_vec(j) <= phase_vec(j-1)
        phase_vec(j) = phase_vec(j) + single(2*pi);
    end
end
end

function a = wrap_to_pi_local(a)
a = mod(a + pi, 2*pi) - pi;
end
