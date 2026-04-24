function [recon, tileReport] = sim_recon_2beam_3d_unified_parallel_batched(raw_data, otf_data, params)
% SIM_RECON_2BEAM_3D_UNIFIED_PARALLEL
% Unified entry point for full-volume or tiled reconstruction.
%
% Canonical dependencies expected on path:
%   make_otf_2beam_3d_canonical.m
%   refit_sim_params_2beam_3d_analytic_canonical.m
%   sim_recon_2beam_3d_core_canonical.m
%
% Tiled mode adds:
%   - optional local tile refit
%   - tile-quality gating to reject unstable refits in featureless/high-uniform tiles
%   - optional disabling of local k0 search on poor tiles
%   - CPU-parallel tile refit
%   - CPU-parallel tile reconstruction when useGPU=false
%   - multi-GPU tile reconstruction when useGPU=true and >1 GPU available
%
% New params:
%   .use_tiling                                  logical, default false
%   .tile_size_xyz                               [ty tx tz]
%   .tile_overlap_xyz                            [oy ox oz], default [0 0 0]
%   .tile_refit                                  logical, default false
%   .tile_refit_accept_threshold_pct             scalar, default 5
%   .tile_auto_accept_refit_if_within_threshold  logical, default true
%   .tile_local_k0_search                        logical, default false
%   .tile_refit_score_min                        scalar, default 0.15
%   .tile_texture_cv_min                         scalar, default 0.02
%   .tile_parallel_refit                         logical, default true
%   .tile_parallel_recon                         logical, default true
%   .tile_parallel_workers                       [], or positive integer
%   .tile_multigpu                               logical, default true
%
% Output:
%   recon      full stitched reconstruction
%   tileReport diagnostics including tile maps and full-volume parameter fields

if ~isfield(params,'use_tiling'), params.use_tiling = false; end
if ~params.use_tiling
    recon = sim_recon_2beam_3d_core(raw_data, otf_data, params);
    tileReport = struct('used_tiling', false);
    return;
end

assert(isfield(params,'tile_size_xyz'), 'params.tile_size_xyz required when use_tiling=true');
if ~isfield(params,'tile_overlap_xyz'), params.tile_overlap_xyz = [0 0 0]; end
if ~isfield(params,'tile_refit'), params.tile_refit = false; end
if ~isfield(params,'tile_refit_accept_threshold_pct'), params.tile_refit_accept_threshold_pct = 5; end
if ~isfield(params,'tile_auto_accept_refit_if_within_threshold'), params.tile_auto_accept_refit_if_within_threshold = true; end
if ~isfield(params,'tile_local_k0_search'), params.tile_local_k0_search = false; end
if ~isfield(params,'tile_refit_score_min'), params.tile_refit_score_min = 0.15; end
if ~isfield(params,'tile_texture_cv_min'), params.tile_texture_cv_min = 0.02; end
if ~isfield(params,'tile_parallel_refit'), params.tile_parallel_refit = true; end
if ~isfield(params,'tile_parallel_recon'), params.tile_parallel_recon = true; end
if ~isfield(params,'tile_parallel_workers'), params.tile_parallel_workers = []; end
if ~isfield(params,'tile_multigpu'), params.tile_multigpu = true; end
if ~isfield(params,'tile_single_gpu_batch'), params.tile_single_gpu_batch = true; end
if ~isfield(params,'tile_gpu_batch_size'), params.tile_gpu_batch_size = []; end
if ~isfield(params,'tile_batch_min_group'), params.tile_batch_min_group = 2; end
if ~isfield(params,'tile_disable_forcemodamp_on_low_texture'), params.tile_disable_forcemodamp_on_low_texture = true; end
if ~isfield(params,'background'), params.background = 0; end
if ~isfield(params,'zoomfact'), params.zoomfact = 2; end
if ~isfield(params,'otfRA'), params.otfRA = true; end
if ~isfield(params,'fastSI'), params.fastSI = false; end
if ~isfield(params,'gammaApo'), params.gammaApo = 1; end
if ~isfield(params,'dampenOrder0'), params.dampenOrder0 = false; end
if ~isfield(params,'suppress_dc'), params.suppress_dc = false; end
if ~isfield(params,'usePerDecomp'), params.usePerDecomp = true; end
if ~isfield(params,'k0_search'), params.k0_search = true; end
if ~isfield(params,'debug'), params.debug = false; end
if ~isfield(params,'phase_step'), params.phase_step = []; end
if ~isfield(params,'phase_vector_rad'), params.phase_vector_rad = []; end
if ~isfield(params,'useGPU'), params.useGPU = []; end
if ~isfield(params,'modamp_thresh'), params.modamp_thresh = 0.05; end
if ~isfield(params,'forcemodamp'), params.forcemodamp = []; end
if ~isfield(params,'pre_resampled_otf'), params.pre_resampled_otf = false; end
if ~isfield(params,'otf_support_thresh'), params.otf_support_thresh = 0.006; end
if ~isfield(params,'sep_cond_warn'), params.sep_cond_warn = 1e4; end
if ~isfield(params,'suppress_band_centers'), params.suppress_band_centers = false; end
if ~isfield(params,'band_suppress_radius_px'), params.band_suppress_radius_px = 6; end
if ~isfield(params,'band_suppress_min_weight'), params.band_suppress_min_weight = 0.05; end
if ~isfield(params,'band_suppress_power'), params.band_suppress_power = 6; end
if ~isfield(params,'use_exact_otf_support'), params.use_exact_otf_support = true; end
if ~isfield(params,'sideband_zcut_factor'), params.sideband_zcut_factor = 1.3; end
if ~isfield(params,'apodizeoutput'), params.apodizeoutput = 1; end
if ~isfield(params,'amp_in_wiener'), params.amp_in_wiener = true; end

[ny, nx, nimgs] = size(raw_data);
nphases = params.nphases;
ndirs = params.ndirs;
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1)==0, 'raw_data size incompatible with nphases*ndirs');
nz = round(nz);

tileSize = round(double(params.tile_size_xyz(:).'));
overlap = round(double(params.tile_overlap_xyz(:).'));
assert(numel(tileSize)==3 && numel(overlap)==3, 'tile_size_xyz and tile_overlap_xyz must be length-3');
tileSize = max([1 1 1], tileSize);
overlap = max([0 0 0], overlap);
assert(all(tileSize - overlap > 0), 'Each tile dimension must exceed its overlap');

% Warn if lateral tiles are too small relative to the stripe period.
stripePeriodPx = min(params.linespacing / params.pxl_dim_data(1), params.linespacing / params.pxl_dim_data(2));
if any(tileSize(1:2) < max(24, 4 * stripePeriodPx))
    warning(['Tile size [%d %d] is small relative to the stripe period (~%.2f px). ' ...
        'This can destabilize local carrier fitting and increase tile artifacts.'], ...
        tileSize(1), tileSize(2), stripePeriodPx);
end

data5d = raw3d_to_5d(raw_data, params.fastSI, nphases, ndirs);
ystarts = make_tile_starts(ny, tileSize(1), overlap(1));
xstarts = make_tile_starts(nx, tileSize(2), overlap(2));
zstarts = make_tile_starts(nz, tileSize(3), overlap(3));
nTy = numel(ystarts); nTx = numel(xstarts); nTz = numel(zstarts);

% Build linear tile job list.
jobs = repmat(struct('iy',[], 'ix',[], 'iz',[], 'y1',[], 'y2',[], 'x1',[], 'x2',[], 'z1',[], 'z2',[], 'tileNy',[], 'tileNx',[], 'tileNz',[], 'key',''), nTy*nTx*nTz, 1);
j = 0;
for iy = 1:nTy
    y1 = ystarts(iy); y2 = min(ny, y1 + tileSize(1) - 1);
    for ix = 1:nTx
        x1 = xstarts(ix); x2 = min(nx, x1 + tileSize(2) - 1);
        for iz = 1:nTz
            z1 = zstarts(iz); z2 = min(nz, z1 + tileSize(3) - 1);
            j = j + 1;
            jobs(j).iy = iy; jobs(j).ix = ix; jobs(j).iz = iz;
            jobs(j).y1 = y1; jobs(j).y2 = y2;
            jobs(j).x1 = x1; jobs(j).x2 = x2;
            jobs(j).z1 = z1; jobs(j).z2 = z2;
            jobs(j).tileNy = y2-y1+1; jobs(j).tileNx = x2-x1+1; jobs(j).tileNz = z2-z1+1;
            jobs(j).key = sprintf('s_%d_%d_%d', jobs(j).tileNy, jobs(j).tileNx, jobs(j).tileNz);
        end
    end
end
nJobs = numel(jobs);

% Precompute/cached resampled OTF by tile size.
uniqueKeys = unique(string({jobs.key}));
otfCache = struct();
for ku = 1:numel(uniqueKeys)
    key = uniqueKeys(ku);
    parts = sscanf(key, 's_%d_%d_%d');
    ty = parts(1); tx = parts(2); tz = parts(3);
    otfCache.(key) = resample_otf_3d_local(otf_data, params.pxl_dim_psf, [ty, tx, tz], params.pxl_dim_data, 2, params.ndirs, params.otfRA);
end

% ----- stage 1: tile analysis / optional local refit (CPU parallel) -----
canPar = can_use_parfor();
refitResults = cell(nJobs,1);

if params.tile_parallel_refit && canPar
    ensure_pool(params.tile_parallel_workers, false);
    parfor j = 1:nJobs
        refitResults{j} = analyze_tile_job(jobs(j), data5d, params);
    end
else
    for j = 1:nJobs
        refitResults{j} = analyze_tile_job(jobs(j), data5d, params);
    end
end

% ----- stage 2: tile reconstruction -----
tileOutputs = cell(nJobs,1);
nGPUs = available_gpu_count();
doMultiGPU = params.tile_parallel_recon && params.useGPU && params.tile_multigpu && nGPUs > 1 && canPar;
doCPUpar = params.tile_parallel_recon && ~params.useGPU && canPar;
doSingleGPUBatch = params.tile_parallel_recon && params.useGPU && ~doMultiGPU && nGPUs == 1 && params.tile_single_gpu_batch;

if doMultiGPU
    ensure_pool(min_nonempty(params.tile_parallel_workers, nGPUs), true);
    parfor j = 1:nJobs
        assign_worker_gpu(nGPUs);
        tileOutputs{j} = reconstruct_tile_job(jobs(j), data5d, otfCache, refitResults{j});
    end
elseif doCPUpar
    ensure_pool(params.tile_parallel_workers, false);
    parfor j = 1:nJobs
        tileOutputs{j} = reconstruct_tile_job(jobs(j), data5d, otfCache, refitResults{j});
    end
elseif doSingleGPUBatch
    tileOutputs = process_single_gpu_batched_jobs(jobs, data5d, otfCache, refitResults, params);
else
    for j = 1:nJobs
        tileOutputs{j} = reconstruct_tile_job(jobs(j), data5d, otfCache, refitResults{j});
    end
end

% ----- stage 3: stitch and diagnostics -----
nyOut = ny * params.zoomfact;
nxOut = nx * params.zoomfact;
recon = zeros(nyOut, nxOut, nz, 'single');

reportList = cell(nTy, nTx, nTz);
tileCenters = nan(nTy, nTx, nTz, 3, 'single');
tileBounds  = nan(nTy, nTx, nTz, 6, 'single');
acceptedAngleDeg = nan(nTy, nTx, nTz, ndirs, 'single');
refinedAngleDeg  = nan(nTy, nTx, nTz, ndirs, 'single');
acceptedLineUm   = nan(nTy, nTx, nTz, ndirs, 'single');
refinedLineUm    = nan(nTy, nTx, nTz, ndirs, 'single');
acceptedPhaseDeg = nan(nTy, nTx, nTz, ndirs, 'single');
refinedPhaseDeg  = nan(nTy, nTx, nTz, ndirs, 'single');
acceptAngleMap   = false(nTy, nTx, nTz, ndirs);
acceptLineMap    = false(nTy, nTx, nTz, ndirs);
acceptPhaseMap   = false(nTy, nTx, nTz, ndirs);
fitScoreMap      = nan(nTy, nTx, nTz, ndirs, 'single');
tileTextureCVMap = nan(nTy, nTx, nTz, 'single');
tileRefitUsedMap = false(nTy, nTx, nTz);

acceptedAngleVol = nan(ny, nx, nz, ndirs, 'single');
refinedAngleVol  = nan(ny, nx, nz, ndirs, 'single');
acceptedLineVol  = nan(ny, nx, nz, ndirs, 'single');
refinedLineVol   = nan(ny, nx, nz, ndirs, 'single');
acceptedPhaseVol = nan(ny, nx, nz, ndirs, 'single');
refinedPhaseVol  = nan(ny, nx, nz, ndirs, 'single');
acceptAngleVolMask = false(ny, nx, nz, ndirs);
acceptLineVolMask  = false(ny, nx, nz, ndirs);
acceptPhaseVolMask = false(ny, nx, nz, ndirs);
fitScoreVol        = nan(ny, nx, nz, ndirs, 'single');
textureCVVol       = nan(ny, nx, nz, 'single');
tileRefitUsedVol   = false(ny, nx, nz);

for j = 1:nJobs
    job = jobs(j);
    out = tileOutputs{j};
    iy = job.iy; ix = job.ix; iz = job.iz;

    recon(out.gy1:out.gy2, out.gx1:out.gx2, out.gz1:out.gz2) = out.cropRecon;

    reportList{iy, ix, iz} = out.localReport;
    tileCenters(iy, ix, iz, :) = single([(job.y1+job.y2)/2, (job.x1+job.x2)/2, (job.z1+job.z2)/2]);
    tileBounds(iy, ix, iz, :) = single([job.y1 job.y2 job.x1 job.x2 job.z1 job.z2]);

    tm = out.tileMaps;
    acceptedAngleDeg(iy, ix, iz, :) = single(tm.angle_deg_accepted);
    refinedAngleDeg(iy, ix, iz, :)  = single(tm.angle_deg_refined);
    acceptedLineUm(iy, ix, iz, :)   = single(tm.linespacing_um_accepted);
    refinedLineUm(iy, ix, iz, :)    = single(tm.linespacing_um_refined);
    acceptedPhaseDeg(iy, ix, iz, :) = single(tm.phase_step_deg_accepted);
    refinedPhaseDeg(iy, ix, iz, :)  = single(tm.phase_step_deg_refined);
    acceptAngleMap(iy, ix, iz, :)   = tm.accept_angle;
    acceptLineMap(iy, ix, iz, :)    = tm.accept_linespacing;
    acceptPhaseMap(iy, ix, iz, :)   = tm.accept_phase;
    fitScoreMap(iy, ix, iz, :)      = single(tm.fit_score);
    tileTextureCVMap(iy, ix, iz)    = single(out.texture_cv);
    tileRefitUsedMap(iy, ix, iz)    = logical(out.refit_used);

    [vly1, vly2, vgy1, vgy2] = compute_blend_indices(job.y1, job.y2, ny, overlap(1), 1, job.tileNy);
    [vlx1, vlx2, vgx1, vgx2] = compute_blend_indices(job.x1, job.x2, nx, overlap(2), 1, job.tileNx);
    [vlz1, vlz2, vgz1, vgz2] = compute_blend_indices(job.z1, job.z2, nz, overlap(3), 1, job.tileNz); %#ok<NASGU>

    for d = 1:ndirs
        acceptedAngleVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d) = single(tm.angle_deg_accepted(d));
        refinedAngleVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d)  = single(tm.angle_deg_refined(d));
        acceptedLineVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d)  = single(tm.linespacing_um_accepted(d));
        refinedLineVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d)   = single(tm.linespacing_um_refined(d));
        acceptedPhaseVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d) = single(tm.phase_step_deg_accepted(d));
        refinedPhaseVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d)  = single(tm.phase_step_deg_refined(d));
        acceptAngleVolMask(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d) = logical(tm.accept_angle(d));
        acceptLineVolMask(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d)  = logical(tm.accept_linespacing(d));
        acceptPhaseVolMask(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d) = logical(tm.accept_phase(d));
        fitScoreVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2, d) = single(tm.fit_score(d));
    end
    textureCVVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2) = single(out.texture_cv);
    tileRefitUsedVol(vgy1:vgy2, vgx1:vgx2, vgz1:vgz2) = logical(out.refit_used);
end

tileReport = struct();
tileReport.used_tiling = true;
tileReport.parallel = struct('used_cpu_parallel_refit', params.tile_parallel_refit && canPar, ...
                             'used_cpu_parallel_recon', doCPUpar, ...
                             'used_multigpu', doMultiGPU, ...
                             'available_gpus', nGPUs);
tileReport.tile_size_xyz = tileSize;
tileReport.tile_overlap_xyz = overlap;
tileReport.tile_grid_size = [nTy nTx nTz];
tileReport.tile_centers_xyz = tileCenters;
tileReport.tile_bounds_xyz = tileBounds;
tileReport.tiles = reportList;
tileReport.accepted_maps = struct('angle_deg', acceptedAngleDeg, ...
                                  'linespacing_um', acceptedLineUm, ...
                                  'phase_step_deg', acceptedPhaseDeg);
tileReport.refined_maps = struct('angle_deg', refinedAngleDeg, ...
                                 'linespacing_um', refinedLineUm, ...
                                 'phase_step_deg', refinedPhaseDeg);
tileReport.accept_masks = struct('angle', acceptAngleMap, ...
                                 'linespacing', acceptLineMap, ...
                                 'phase', acceptPhaseMap);
tileReport.fit_score = fitScoreMap;
tileReport.texture_cv = tileTextureCVMap;
tileReport.tile_refit_used = tileRefitUsedMap;
tileReport.volume_fields = struct( ...
    'accepted', struct('angle_deg', acceptedAngleVol, ...
                       'linespacing_um', acceptedLineVol, ...
                       'phase_step_deg', acceptedPhaseVol), ...
    'refined',  struct('angle_deg', refinedAngleVol, ...
                       'linespacing_um', refinedLineVol, ...
                       'phase_step_deg', refinedPhaseVol), ...
    'accept_masks', struct('angle', acceptAngleVolMask, ...
                           'linespacing', acceptLineVolMask, ...
                           'phase', acceptPhaseVolMask), ...
    'fit_score', fitScoreVol, ...
    'texture_cv', textureCVVol, ...
    'tile_refit_used', tileRefitUsedVol);
end

% ---------------- analysis and reconstruction helpers -----------------

function result = analyze_tile_job(job, data5d, params)
tile5d = data5d(job.y1:job.y2, job.x1:job.x2, job.z1:job.z2, :, :);
rawTile = raw5d_to_3d(tile5d, params.fastSI);

localParams = ensure_recon_optional_defaults(params);
localParams.use_tiling = false;
localParams.pre_resampled_otf = true;
localParams.k0_search = false;  % robust default for tiles; only enable below if requested and tile is informative
if params.tile_disable_forcemodamp_on_low_texture && ~isempty(localParams.forcemodamp)
    localParams.forcemodamp = [];
end

result = struct();
result.localReport = struct('bounds_xyz', [job.y1 job.y2 job.x1 job.x2 job.z1 job.z2], ...
                            'size_xyz', [job.tileNy job.tileNx job.tileNz]);

% Basic texture metric: coefficient of variation of a mean-projection after background subtraction.
avgProj = mean(rawTile, 3);
mu = mean(avgProj(:), 'omitnan');
sigma = std(avgProj(:), 0, 'omitnan');
texture_cv = sigma / max(abs(mu), eps('single'));
result.texture_cv = single(texture_cv);

if params.tile_refit && texture_cv >= params.tile_texture_cv_min
    [paramsCand, repRefit, tileMapsCand] = apply_local_refit_base(rawTile, localParams);
    fitScore = tileMapsCand.fit_score;
    goodScore = any(isfinite(fitScore)) && all(fitScore(isfinite(fitScore)) >= params.tile_refit_score_min);

    if goodScore
        localParams = paramsCand;
        result.localReport.refit = repRefit;
        result.tileMaps = tileMapsCand;
        result.refit_used = true;
        if params.tile_local_k0_search
            localParams.k0_search = true;
        end
    else
        [result.localReport.refit, result.tileMaps] = build_no_refit_tile_maps(localParams);
        result.localReport.refit.rejected_reason = 'low_fit_score';
        result.refit_used = false;
    end
else
    [result.localReport.refit, result.tileMaps] = build_no_refit_tile_maps(localParams);
    if params.tile_refit
        result.localReport.refit.rejected_reason = 'low_texture';
    end
    result.refit_used = false;
end

result.localParams = localParams;
end

function out = reconstruct_tile_job(job, data5d, otfCache, analysisResult)
tile5d = data5d(job.y1:job.y2, job.x1:job.x2, job.z1:job.z2, :, :);
rawTile = raw5d_to_3d(tile5d, analysisResult.localParams.fastSI);
otfTile = otfCache.(job.key);
reconTile = sim_recon_2beam_3d_core(rawTile, otfTile, analysisResult.localParams);

[ly1, ly2, gy1, gy2] = compute_blend_indices(job.y1, job.y2, size(data5d,1), analysisResult.localParams.tile_overlap_xyz(1), analysisResult.localParams.zoomfact, size(reconTile,1));
[lx1, lx2, gx1, gx2] = compute_blend_indices(job.x1, job.x2, size(data5d,2), analysisResult.localParams.tile_overlap_xyz(2), analysisResult.localParams.zoomfact, size(reconTile,2));
[lz1, lz2, gz1, gz2] = compute_blend_indices(job.z1, job.z2, size(data5d,3), analysisResult.localParams.tile_overlap_xyz(3), 1, size(reconTile,3));

out = struct();
out.cropRecon = reconTile(ly1:ly2, lx1:lx2, lz1:lz2);
out.gy1 = gy1; out.gy2 = gy2;
out.gx1 = gx1; out.gx2 = gx2;
out.gz1 = gz1; out.gz2 = gz2;
out.localReport = analysisResult.localReport;
out.tileMaps = analysisResult.tileMaps;
out.texture_cv = analysisResult.texture_cv;
out.refit_used = analysisResult.refit_used;
end

function [paramsOut, rep, tileMaps] = apply_local_refit_base(rawTile, paramsIn)
paramsOut = paramsIn;
rep = struct();

fitParams = struct();
fitParams.nphases = paramsIn.nphases;
fitParams.ndirs = paramsIn.ndirs;
fitParams.pxl_dim_data = paramsIn.pxl_dim_data;
fitParams.k0angles = paramsIn.k0angles;
fitParams.linespacing = paramsIn.linespacing;
fitParams.background = paramsIn.background;
fitParams.fastSI = paramsIn.fastSI;
if isfield(paramsIn,'otf_support_thresh') && ~isempty(paramsIn.otf_support_thresh)
    fitParams.otf_support_thresh = paramsIn.otf_support_thresh;
else
    fitParams.otf_support_thresh = 0.006;
end
if isfield(paramsIn,'phase_step'), fitParams.phase_step = paramsIn.phase_step; end
if isfield(paramsIn,'phase_vector_rad'), fitParams.phase_vector_rad = paramsIn.phase_vector_rad; end
if isfield(paramsIn,'refit_k0_search_window_px'), fitParams.refit_k0_search_window_px = paramsIn.refit_k0_search_window_px; end
if isfield(paramsIn,'refit_apodize_xy_px'), fitParams.refit_apodize_xy_px = paramsIn.refit_apodize_xy_px; end

fitReport = refit_sim_params_2beam_3d(rawTile, fitParams);
rep.fitReport = fitReport;

proposedPhaseVectors = build_proposed_phase_vectors(paramsIn);
refinedPhaseVectors = double(fitReport.phase_vectors_rad);
proposedPhaseStep = effective_phase_step_rad(proposedPhaseVectors);
refinedPhaseStep = effective_phase_step_rad(refinedPhaseVectors);

k0angles_refined = double(fitReport.k0angles_refined(:)).';
linespacing_refined = double(fitReport.linespacing_refined(:)).';
angle_dev_deg = abs(rad2deg(wrap_to_pi_local(k0angles_refined - paramsIn.k0angles)));
angle_dev_pct = 100 * angle_dev_deg / 180;
linespacing_dev_pct = 100 * abs((linespacing_refined - paramsIn.linespacing) ./ max(abs(paramsIn.linespacing), eps));
phase_step_dev_pct = 100 * abs((refinedPhaseStep - proposedPhaseStep) ./ max(abs(proposedPhaseStep), eps));

thr = paramsIn.tile_refit_accept_threshold_pct;
if paramsIn.tile_auto_accept_refit_if_within_threshold
    accept_angle = angle_dev_pct <= thr;
    accept_linespacing = linespacing_dev_pct <= thr;
    accept_phase = phase_step_dev_pct <= thr;
else
    accept_angle = true(1, paramsIn.ndirs);
    accept_linespacing = true(1, paramsIn.ndirs);
    accept_phase = true(1, paramsIn.ndirs);
end

k0angles_acc = paramsIn.k0angles;
k0angles_acc(accept_angle) = k0angles_refined(accept_angle);
linespacing_acc = repmat(paramsIn.linespacing, 1, paramsIn.ndirs);
linespacing_acc(accept_linespacing) = linespacing_refined(accept_linespacing);
phase_vectors_acc = proposedPhaseVectors;
phase_vectors_acc(accept_phase, :) = refinedPhaseVectors(accept_phase, :);

paramsOut.k0angles = k0angles_acc;
paramsOut.linespacing = mean(linespacing_acc);
paramsOut.phase_step = [];
paramsOut.phase_vector_rad = phase_vectors_acc;

rep.acceptance = struct('threshold_pct', thr, ...
    'angle_dev_pct', angle_dev_pct, 'linespacing_dev_pct', linespacing_dev_pct, 'phase_step_dev_pct', phase_step_dev_pct, ...
    'accept_angle', accept_angle, 'accept_linespacing', accept_linespacing, 'accept_phase', accept_phase, ...
    'k0angles_accepted', k0angles_acc, 'linespacing_accepted', linespacing_acc, 'phase_vectors_accepted', phase_vectors_acc);

tileMaps = struct();
tileMaps.angle_deg_refined = rad2deg(k0angles_refined);
tileMaps.angle_deg_accepted = rad2deg(k0angles_acc);
tileMaps.linespacing_um_refined = linespacing_refined;
tileMaps.linespacing_um_accepted = linespacing_acc;
tileMaps.phase_step_deg_refined = rad2deg(refinedPhaseStep);
tileMaps.phase_step_deg_accepted = rad2deg(effective_phase_step_rad(phase_vectors_acc));
tileMaps.accept_angle = accept_angle;
tileMaps.accept_linespacing = accept_linespacing;
tileMaps.accept_phase = accept_phase;
if isfield(fitReport, 'directions')
    fit_score = nan(1, paramsIn.ndirs);
    for d = 1:min(numel(fitReport.directions), paramsIn.ndirs)
        if isfield(fitReport.directions(d), 'score')
            fit_score(d) = fitReport.directions(d).score;
        end
    end
    tileMaps.fit_score = fit_score;
else
    tileMaps.fit_score = nan(1, paramsIn.ndirs);
end
end


% ---------------- single-GPU batched helpers ----------------

function tileOutputs = process_single_gpu_batched_jobs(jobs, data5d, otfCache, refitResults, params)
nJobs = numel(jobs);
tileOutputs = cell(nJobs,1);

sigMap = containers.Map('KeyType','char','ValueType','any');
sigList = cell(nJobs,1);
for j = 1:nJobs
    sig = tile_batch_signature(jobs(j), refitResults{j}, params);
    sigList{j} = sig;
    if ~isKey(sigMap, sig)
        sigMap(sig) = j;
    else
        sigMap(sig) = [sigMap(sig), j];
    end
end

allKeys = keys(sigMap);
for ik = 1:numel(allKeys)
    key = allKeys{ik};
    idxs = sigMap(key);
    if startsWith(key, 'SERIAL|') || numel(idxs) < params.tile_batch_min_group
        for ii = 1:numel(idxs)
            j = idxs(ii);
            tileOutputs{j} = reconstruct_tile_job(jobs(j), data5d, otfCache, refitResults{j});
        end
        continue;
    end

    batchSize = choose_single_gpu_batch_size(jobs(idxs), refitResults{idxs(1)}.localParams, params);
    for s = 1:batchSize:numel(idxs)
        take = idxs(s:min(s+batchSize-1, numel(idxs)));
        outs = reconstruct_tile_group_single_gpu(jobs(take), data5d, otfCache, refitResults(take));
        for kk = 1:numel(take)
            tileOutputs{take(kk)} = outs{kk};
        end
    end
end
end

function sig = tile_batch_signature(job, analysisResult, params)
lp = ensure_recon_optional_defaults(analysisResult.localParams);
if lp.k0_search
    sig = ['SERIAL|', job.key, '|k0search'];
    return;
end
sig = sprintf(['BATCH|%s|zf%.6g|bg%.6g|ls%.6g|ka%s|pv%s|w%.6g|ga%.6g|d0%d|dc%d|pd%d|fm%s|sb%d|br%.6g|bm%.6g|bp%.6g|ex%d|zf1%.6g|apo%d|ampw%d'], ...
    job.key, lp.zoomfact, lp.background, lp.linespacing, ...
    mat2str(round(double(lp.k0angles(:)).', 8)), ...
    mat2str(round(double(resolve_phase_vector_signature(lp)), 8)), ...
    lp.wiener, lp.gammaApo, lp.dampenOrder0, lp.suppress_dc, lp.usePerDecomp, ...
    mat2str(round(double(resolve_forcemodamp_signature(lp)), 8)), ...
    lp.suppress_band_centers, lp.band_suppress_radius_px, lp.band_suppress_min_weight, lp.band_suppress_power, ...
    lp.use_exact_otf_support, lp.sideband_zcut_factor, lp.apodizeoutput, lp.amp_in_wiener);
end

function pv = resolve_phase_vector_signature(lp)
if isfield(lp,'phase_vector_rad') && ~isempty(lp.phase_vector_rad)
    pv = lp.phase_vector_rad;
elseif isfield(lp,'phase_step') && ~isempty(lp.phase_step)
    delta_phi = (lp.phase_step * 1e-3) / lp.linespacing * 2*pi;
    pv = [0, delta_phi, 2*delta_phi];
else
    pv = [0, 2*pi/3, 4*pi/3];
end
end

function fm = resolve_forcemodamp_signature(lp)
if isfield(lp,'forcemodamp') && ~isempty(lp.forcemodamp)
    fm = lp.forcemodamp;
else
    fm = [];
end
end

function batchSize = choose_single_gpu_batch_size(jobGroup, localParams, params)
if ~isempty(params.tile_gpu_batch_size)
    batchSize = max(1, round(params.tile_gpu_batch_size));
    return;
end
try
    g = gpuDevice();
    avail = double(g.AvailableMemory);
catch
    batchSize = 1;
    return;
end

tileNy = jobGroup(1).tileNy;
tileNx = jobGroup(1).tileNx;
tileNz = jobGroup(1).tileNz;
nyOut = tileNy * localParams.zoomfact;
nxOut = tileNx * localParams.zoomfact;
nimgs = tileNz * localParams.nphases * localParams.ndirs;

% Conservative byte estimate per tile for the batched path.
bytesRaw   = double(tileNy * tileNx * nimgs) * 4;
bytesRecon = double(nyOut * nxOut * tileNz) * 8;   % complex single
bytesBands = double(nyOut * nxOut * tileNz) * 8 * 8; % multiple complex intermediates
bytesPerTile = bytesRaw + bytesRecon + bytesBands;

target = 0.35 * avail;
batchSize = max(1, floor(target / max(bytesPerTile, 1)));
batchSize = min(batchSize, numel(jobGroup));
end

function outs = reconstruct_tile_group_single_gpu(jobGroup, data5d, otfCache, analysisGroup)
nB = numel(jobGroup);
job0 = jobGroup(1);
lp = analysisGroup{1}.localParams;

rawBatch = zeros(job0.tileNy, job0.tileNx, job0.tileNz * lp.nphases * lp.ndirs, nB, 'single');
for b = 1:nB
    j = jobGroup(b);
    tile5d = data5d(j.y1:j.y2, j.x1:j.x2, j.z1:j.z2, :, :);
    rawBatch(:,:,:,b) = raw5d_to_3d(tile5d, lp.fastSI);
end

otfTile = otfCache.(job0.key);
reconBatch = sim_recon_2beam_3d_core_batched_shared(rawBatch, otfTile, lp);

outs = cell(nB,1);
for b = 1:nB
    j = jobGroup(b);
    ar = analysisGroup{b};
    [ly1, ly2, gy1, gy2] = compute_blend_indices(j.y1, j.y2, size(data5d,1), lp.tile_overlap_xyz(1), lp.zoomfact, size(reconBatch,1));
    [lx1, lx2, gx1, gx2] = compute_blend_indices(j.x1, j.x2, size(data5d,2), lp.tile_overlap_xyz(2), lp.zoomfact, size(reconBatch,2));
    [lz1, lz2, gz1, gz2] = compute_blend_indices(j.z1, j.z2, size(data5d,3), lp.tile_overlap_xyz(3), 1, size(reconBatch,3));

    out = struct();
    out.cropRecon = reconBatch(ly1:ly2, lx1:lx2, lz1:lz2, b);
    out.gy1 = gy1; out.gy2 = gy2;
    out.gx1 = gx1; out.gx2 = gx2;
    out.gz1 = gz1; out.gz2 = gz2;
    out.localReport = ar.localReport;
    out.tileMaps = ar.tileMaps;
    out.texture_cv = ar.texture_cv;
    out.refit_used = ar.refit_used;
    outs{b} = out;
end
end

% ---------------- misc helpers ----------------

function phase_vectors = build_proposed_phase_vectors(params)
ndirs = params.ndirs;
if isfield(params,'phase_vector_rad') && ~isempty(params.phase_vector_rad)
    pv = params.phase_vector_rad;
    if isvector(pv)
        phase_vectors = repmat(reshape(pv,1,[]), ndirs, 1);
    else
        phase_vectors = pv;
    end
elseif isfield(params,'phase_step') && ~isempty(params.phase_step)
    phase_vectors = zeros(ndirs, params.nphases);
    phase_step_nm = params.phase_step;
    if isscalar(phase_step_nm)
        phase_step_nm = repmat(phase_step_nm, 1, ndirs);
    end
    for d = 1:ndirs
        delta_phi = (phase_step_nm(d) * 1e-3) / params.linespacing * 2*pi;
        phase_vectors(d,:) = (0:params.nphases-1) * delta_phi;
    end
else
    base = (0:params.nphases-1) * (2*pi / params.nphases);
    phase_vectors = repmat(base, ndirs, 1);
end
end

function step_rad = effective_phase_step_rad(phase_vectors_rad)
step_rad = mean(diff(unwrap(phase_vectors_rad, [], 2), 1, 2), 2).';
end

function a = wrap_to_pi_local(a)
a = mod(a + pi, 2*pi) - pi;
end


function OTF_out = resample_otf_3d_local(otf_in, pxl_psf, data_dims, pxl_data, norders, ndirs, otfRA)
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
for d = 1:n_dirs_out
    for ord = 1:norders
        otf_vol = otf_in(:,:,:,ord,min(d,ndirs_otf));
        OTF_out(:,:,:,ord,d) = single(interp3(OTF_XX, OTF_YY, OTF_ZZ, ...
            otf_vol, MAP_XX, MAP_YY, MAP_ZZ, 'cubic', 0+0i));
    end
end
end

function tf = can_use_parfor()
tf = ~isempty(ver('parallel')) && license('test','Distrib_Computing_Toolbox');
end

function ensure_pool(numWorkers, useProcessPool)
if isempty(gcp('nocreate'))
    try
        if nargin < 2, useProcessPool = false; end
        if isempty(numWorkers)
            if useProcessPool
                parpool('local');
            else
                parpool('threads');
            end
        else
            if useProcessPool
                parpool('local', numWorkers);
            else
                parpool('threads', numWorkers);
            end
        end
    catch
        % If pool creation fails, fall back silently to serial execution in caller.
    end
end
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

function v = min_nonempty(a, b)
if isempty(a)
    v = b;
else
    v = min(a, b);
end
end



function p = ensure_recon_optional_defaults(p)
if ~isfield(p,'background'), p.background = 0; end
if ~isfield(p,'zoomfact'), p.zoomfact = 2; end
if ~isfield(p,'otfRA'), p.otfRA = true; end
if ~isfield(p,'fastSI'), p.fastSI = false; end
if ~isfield(p,'gammaApo'), p.gammaApo = 1; end
if ~isfield(p,'dampenOrder0'), p.dampenOrder0 = false; end
if ~isfield(p,'suppress_dc'), p.suppress_dc = false; end
if ~isfield(p,'usePerDecomp'), p.usePerDecomp = true; end
if ~isfield(p,'k0_search'), p.k0_search = true; end
if ~isfield(p,'debug'), p.debug = false; end
if ~isfield(p,'phase_step'), p.phase_step = []; end
if ~isfield(p,'phase_vector_rad'), p.phase_vector_rad = []; end
if ~isfield(p,'useGPU'), p.useGPU = []; end
if ~isfield(p,'modamp_thresh'), p.modamp_thresh = 0.05; end
if ~isfield(p,'forcemodamp'), p.forcemodamp = []; end
if ~isfield(p,'pre_resampled_otf'), p.pre_resampled_otf = false; end
if ~isfield(p,'otf_support_thresh'), p.otf_support_thresh = 0.006; end
if ~isfield(p,'sep_cond_warn'), p.sep_cond_warn = 1e4; end
if ~isfield(p,'suppress_band_centers'), p.suppress_band_centers = false; end
if ~isfield(p,'band_suppress_radius_px'), p.band_suppress_radius_px = 6; end
if ~isfield(p,'band_suppress_min_weight'), p.band_suppress_min_weight = 0.05; end
if ~isfield(p,'band_suppress_power'), p.band_suppress_power = 6; end
if ~isfield(p,'use_exact_otf_support'), p.use_exact_otf_support = true; end
if ~isfield(p,'sideband_zcut_factor'), p.sideband_zcut_factor = 1.3; end
if ~isfield(p,'apodizeoutput'), p.apodizeoutput = 1; end
if ~isfield(p,'amp_in_wiener'), p.amp_in_wiener = true; end
end

% ---------------- missing local helpers restored ----------------

function data5d = raw3d_to_5d(raw_data, fastSI, nphases, ndirs)
[ny, nx, nimgs] = size(raw_data);
nz = nimgs / (nphases * ndirs);
assert(mod(nz,1) == 0, 'raw_data size incompatible with nphases*ndirs');
nz = round(nz);
if fastSI
    data5d = reshape(raw_data, ny, nx, nphases, ndirs, nz);
    data5d = permute(data5d, [1 2 5 3 4]); % [ny nx nz phase dir]
else
    data5d = reshape(raw_data, ny, nx, nphases, nz, ndirs);
    data5d = permute(data5d, [1 2 4 3 5]); % [ny nx nz phase dir]
end
end

function raw3d = raw5d_to_3d(data5d, fastSI)
[ny, nx, ~, ~, ~] = size(data5d);
if fastSI
    tmp = permute(data5d, [1 2 4 5 3]);
else
    tmp = permute(data5d, [1 2 4 3 5]);
end
raw3d = reshape(tmp, ny, nx, []);
end

function starts = make_tile_starts(N, tileN, overlapN)
if tileN >= N
    starts = 1;
    return;
end
step = tileN - overlapN;
starts = 1:step:(N - tileN + 1);
if starts(end) ~= (N - tileN + 1)
    starts = [starts, (N - tileN + 1)];
end
starts = unique(starts);
end

function [l1, l2, g1, g2] = compute_blend_indices(startIdx, endIdx, fullN, overlapN, zoom, localOutN)
outStart = (startIdx - 1) * zoom + 1;
outEnd   = endIdx * zoom;
outFullN = fullN * zoom;
outOverlap = overlapN * zoom;

leftTrim = 0;
rightTrim = 0;

if outStart > 1
    leftTrim = floor(outOverlap / 2);
end
if outEnd < outFullN
    rightTrim = outOverlap - floor(outOverlap / 2);
end

leftTrim = min(leftTrim, localOutN - 1);
rightTrim = min(rightTrim, localOutN - 1 - leftTrim);

l1 = 1 + leftTrim;
l2 = localOutN - rightTrim;
g1 = outStart + leftTrim;
g2 = outEnd - rightTrim;
end

function [rep, tileMaps] = build_no_refit_tile_maps(localParams)
rep = struct();
phase_vectors = build_proposed_phase_vectors(localParams);
phase_step_deg = rad2deg(effective_phase_step_rad(phase_vectors));

tileMaps = struct();
tileMaps.angle_deg_refined       = rad2deg(localParams.k0angles);
tileMaps.angle_deg_accepted      = rad2deg(localParams.k0angles);
tileMaps.linespacing_um_refined  = repmat(localParams.linespacing, 1, localParams.ndirs);
tileMaps.linespacing_um_accepted = repmat(localParams.linespacing, 1, localParams.ndirs);
tileMaps.phase_step_deg_refined  = phase_step_deg;
tileMaps.phase_step_deg_accepted = phase_step_deg;
tileMaps.accept_angle            = true(1, localParams.ndirs);
tileMaps.accept_linespacing      = true(1, localParams.ndirs);
tileMaps.accept_phase            = true(1, localParams.ndirs);
tileMaps.fit_score               = nan(1, localParams.ndirs);
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