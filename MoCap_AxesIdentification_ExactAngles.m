%% =========================================================================
%  Ankle Rotation Axis Identification — Exact Angle Decomposition
%  Vicon ASCII CSV @ 100 Hz | Subjects: Lechen, Tianyu
%
%  COPY OF ankleAxesIdentification_CentredPCA.m
%  ONLY CHANGE: angle decomposition method (Figures 4, 5, 6, 7).
%
%  The centred PCA axis identification pipeline is IDENTICAL to the
%  original. Only the per-frame angle computation is replaced:
%
%    LINEAR (original):   theta = dot(rv, n)         [BCH approximation]
%    EXACT  (this file):  decomposeAnglesExact(R_rel, n_PF, n_IE)
%
%  METHOD — Sequential floating-axis (Grood-Suntay / JCS):
%    Solves exactly:  R(n_PF, theta_PF) * R(n_IE_body, theta_IE) = R_rel
%    Convention: PF/DF is proximal (calf-fixed) axis, applied first.
%                IE/EV is distal  (foot-fixed)  axis, applied second.
%    This matches the OpenSim model joint convention.
%
%    Closed-form, no iteration. Valid for any ROM.
%    Singularity only at theta_IE = +-90 deg (anatomically unreachable).
%
%  Figure 4 shows BOTH exact (solid) and linear (dashed) to quantify
%  the BCH error at large angles.
%
%  CALF FRAME CONVENTION (ISB/OpenSim tibia_l):
%    X = anterior  (calfVec x Z)
%    Y = superior  (Z x X)
%    Z = lateral   (Knee2 - Knee1)
%
%  FIGURES:
%    1. Rotation vector clouds + identified axes (PF and IE, 2x2 grid)
%    2. Foot marker arcs in calf frame (PF vs IE trials)
%    3. Identified axes + joint centre in 3D (calf frame)
%    4. PF/DF and Inv/Ev angle time series — EXACT (solid) + LINEAR (dashed)
%    5. Live animation of identified axes (if SHOW_ANIMATION = true)
%    6. Joint angle space diagnostic (exact angles)
%    7. DoF dimensionality test (scree, PC3 vs time, residual analysis)
%
% =========================================================================

clc; clear; close all;

%% -------------------------------------------------------------------------
%  USER PARAMETERS
% -------------------------------------------------------------------------
SUBJECT_NAME  = 'Tianyu';          % 'Lechen' or 'Tianyu'
PF_TRIALS     = {'FE01','FE02'};   % predominantly PF/DF trials
IE_TRIALS     = {'IE01','IE02'};   % predominantly Inv/Ev trials
NEUTRAL_TRIAL = 'FE01';            % trial used to find neutral frame

MIN_ANGLE_DEG     = 2.0;    % skip frames with total rotation < this (noise)
MAX_ANGLE_DEG_POS = 60.0;   % rotation vector collection ceiling (degrees)

BASE_DIR = '/Users/rv315/Downloads/Ankle Exo Optimisation/MoCap Data/AxesIdentification12032026';

%% -------------------------------------------------------------------------
%  USER PARAMETERS — Animation (Figure 5)
% -------------------------------------------------------------------------
SHOW_ANIMATION = false;        % set false to skip animation entirely
VIS_TRIAL      = 'Random01';  % trial to animate
VIS_FRAME_SKIP = 10;          % render every Nth frame (1 = every frame)
VIS_SPEED      = 1.0;         % playback speed multiplier
VIS_MODE       = false;
AXIS_SCALE     = 70;          % mm — drawn axis length
SHOW_LABELS    = false;        % show marker name labels in animation


%% -------------------------------------------------------------------------
%  LOAD REFERENCE TRIAL — get marker names & column layout
% -------------------------------------------------------------------------
refPath = csvPathFor(BASE_DIR, SUBJECT_NAME, NEUTRAL_TRIAL);
[markerNames, markerXCols, ~, ~] = loadTrialCSV(refPath);

iAnkle3 = find(strcmp(markerNames,'Ankle3'));
iAnkle5 = find(strcmp(markerNames,'Ankle5'));
iKnee1  = find(strcmp(markerNames,'Knee1'));
iKnee2  = find(strcmp(markerNames,'Knee2'));
iCalfU  = find(~cellfun(@isempty, regexp(markerNames,'^CalfU\d+$')));
iCalfL  = find(~cellfun(@isempty, regexp(markerNames,'^CalfL\d+$')));
iFoot   = find(~cellfun(@isempty, regexp(markerNames,'^Foot\d+$')));
assert(~isempty(iAnkle3)&&~isempty(iAnkle5), 'Ankle3/5 not found');
assert(~isempty(iKnee1) &&~isempty(iKnee2),  'Knee1/2 not found');
assert(~isempty(iCalfU) &&~isempty(iCalfL),  'CalfU/L not found');
assert(~isempty(iFoot),                       'Foot markers not found');

GROUP_COLOURS = struct( ...
    'Knee',  [0.20 0.60 1.00], 'CalfU', [0.10 0.80 0.40], ...
    'CalfL', [0.00 0.55 0.25], 'Ankle', [1.00 0.75 0.00], ...
    'Foot',  [1.00 0.30 0.15]);
DEFAULT_COLOUR = [0.70 0.70 0.70];
nMarkers = numel(markerNames);
markerColours = zeros(nMarkers, 3);
for m = 1:nMarkers
    grp = regexprep(markerNames{m}, '\d+$', '');
    if isfield(GROUP_COLOURS, grp)
        markerColours(m,:) = GROUP_COLOURS.(grp);
    else
        markerColours(m,:) = DEFAULT_COLOUR;
    end
end

%% -------------------------------------------------------------------------
%  FIND NEUTRAL FRAME — global search across ALL available trials
% -------------------------------------------------------------------------
NEUTRAL_SEARCH_TRIALS = { ...
    'Static01','Static02', ...
    'FE01','FE02', ...
    'IE01','IE02', ...
    'FEIE01','FEIE02', ...
    'IEFE01','IEFE02', ...
    'Random01','Random02','Random03'};

fprintf('\n--- Searching for best neutral frame across ALL trials ---\n');
bestNeutralScore = -Inf;
iNeutral         = 1;
neutralTrial     = NEUTRAL_TRIAL;
R_calf_n         = eye(3);
R_foot_n         = eye(3);
O_calf_n         = zeros(1,3);
mdN_best         = [];

for nti = 1:numel(NEUTRAL_SEARCH_TRIALS)
    tNameN  = NEUTRAL_SEARCH_TRIALS{nti};
    tDirN   = fullfile(BASE_DIR, SUBJECT_NAME, tNameN);
    tFilesN = dir(fullfile(tDirN,'*.csv'));
    if isempty(tFilesN), continue; end
    tPathN  = fullfile(tDirN, tFilesN(1).name);

    md_t = loadTrialData(tPathN, markerXCols);
    [O_c_t, R_c_t, calfV_t] = computeCalfFrames(md_t, ...
        iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
    [~, R_f_t, footN_t] = computeFootFrames(md_t, iFoot, calfV_t);

    scores_t = dot(calfV_t, footN_t, 2);
    [sc_t, fi_t] = max(scores_t);
    fprintf('  %-12s  best frame %4d / %4d   alignment = %.5f\n', ...
        tNameN, fi_t, size(md_t,1), sc_t);

    if sc_t > bestNeutralScore
        bestNeutralScore = sc_t;
        iNeutral         = fi_t;
        neutralTrial     = tNameN;
        R_calf_n         = R_c_t(:,:,fi_t);
        R_foot_n         = R_f_t(:,:,fi_t);
        O_calf_n         = O_c_t(fi_t,:);
        mdN_best         = md_t;
    end
end

R_correct = R_foot_n' * R_calf_n;
fprintf('\n  >>> BEST NEUTRAL: trial = %-12s  frame = %d  alignment = %.5f\n', ...
    neutralTrial, iNeutral, bestNeutralScore);

% Extract ALL markers at the neutral frame
p0_allMarkers_world  = nan(nMarkers, 3);
p0_allMarkers_inCalf = nan(nMarkers, 3);
for m = 1:nMarkers
    pw = squeeze(mdN_best(iNeutral, m, :))';
    if any(isnan(pw)), continue; end
    p0_allMarkers_world(m,:)  = pw;
    p0_allMarkers_inCalf(m,:) = (R_calf_n' * (pw(:) - O_calf_n(:)))';
end

p0_foot_world  = p0_allMarkers_world(iFoot, :);
p0_foot_all    = p0_allMarkers_inCalf(iFoot, :);
pAnkle3_world  = p0_allMarkers_world(iAnkle3, :);
pAnkle5_world  = p0_allMarkers_world(iAnkle5, :);
pAnkle3_inCalf = p0_allMarkers_inCalf(iAnkle3, :);
pAnkle5_inCalf = p0_allMarkers_inCalf(iAnkle5, :);

validFoot_w              = p0_foot_world(all(~isnan(p0_foot_world), 2), :);
foot_svd_centroid_world  = mean(validFoot_w, 1);
foot_svd_centroid_inCalf = (R_calf_n' * (foot_svd_centroid_world(:) - O_calf_n(:)))';

clear mdN_best;


%% -------------------------------------------------------------------------
%  COLLECT ROTATION VECTORS + BUILD LEAST-SQUARES MATRICES
% -------------------------------------------------------------------------
rotVecs_PF = [];  rotVecs_IE = [];
Amat_PF = [];  bvec_PF = [];
Amat_IE = [];  bvec_IE = [];

trialSets = {PF_TRIALS, IE_TRIALS};

for setIdx = 1:2
    for t = 1:numel(trialSets{setIdx})
        tName = trialSets{setIdx}{t};
        fprintf('Processing %s ...\n', tName);
        md = loadTrialData(csvPathFor(BASE_DIR, SUBJECT_NAME, tName), markerXCols);
        nF = size(md,1);
        [O_c, R_c, calfV] = computeCalfFrames(md, iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
        [~,   R_f, ~]     = computeFootFrames(md, iFoot, calfV);

        for f = 1:nF
            R_rel   = R_c(:,:,f)' * R_f(:,:,f) * R_correct;
            rv      = rotmat2vec(R_rel);
            if isempty(rv), continue; end
            ang_deg = norm(rv) * 180/pi;
            if ang_deg < MIN_ANGLE_DEG || ang_deg > MAX_ANGLE_DEG_POS, continue; end

            if setIdx == 1
                rotVecs_PF(end+1,:) = rv; %#ok<SAGROW>
            else
                rotVecs_IE(end+1,:) = rv; %#ok<SAGROW>
            end

            Oc = O_c(f,:);  Rc = R_c(:,:,f);
            for k = 1:numel(iFoot)
                if any(isnan(p0_foot_all(k,:))), continue; end
                pw = squeeze(md(f, iFoot(k), :))';
                if any(isnan(pw)), continue; end
                p_f     = (Rc' * (pw - Oc)')';
                p_0     = p0_foot_all(k,:);
                A_block = eye(3) - R_rel;
                b_block = (p_f - (R_rel * p_0')')';
                if setIdx == 1
                    Amat_PF = [Amat_PF; A_block]; %#ok<AGROW>
                    bvec_PF = [bvec_PF; b_block]; %#ok<AGROW>
                else
                    Amat_IE = [Amat_IE; A_block]; %#ok<AGROW>
                    bvec_IE = [bvec_IE; b_block]; %#ok<AGROW>
                end
            end
        end
    end
end

fprintf('\nPF cloud: %d frames  |  IE cloud: %d frames\n', ...
    size(rotVecs_PF,1), size(rotVecs_IE,1));


%% -------------------------------------------------------------------------
%  STEP 3 — AXIS DIRECTIONS: Centred PCA
% -------------------------------------------------------------------------
mu_PF     = mean(rotVecs_PF, 1);
rvc_PF    = rotVecs_PF - mu_PF;
[~,S_PF,V_PF] = svd(rvc_PF, 'econ');
n_PF      = V_PF(:,1)';
if n_PF(3) < 0, n_PF = -n_PF; end
sv_PF     = diag(S_PF);
varexp_PF = sv_PF(1)^2 / sum(sv_PF.^2) * 100;

mu_IE     = mean(rotVecs_IE, 1);
rvc_IE    = rotVecs_IE - mu_IE;
[~,S_IE,V_IE] = svd(rvc_IE, 'econ');
n_IE      = V_IE(:,1)';
if n_IE(1) < 0, n_IE = -n_IE; end
sv_IE     = diag(S_IE);
varexp_IE = sv_IE(1)^2 / sum(sv_IE.^2) * 100;

axesAngle_deg = acos(min(1, abs(dot(n_PF, n_IE)))) * 180/pi;

fprintf('\n=== STEP 3: AXIS DIRECTIONS (calf frame, centred PCA) ===\n');
fprintf('  n_PF = [%+.4f %+.4f %+.4f]  PC1 = %.1f%%  N = %d frames\n', ...
    n_PF, varexp_PF, size(rotVecs_PF,1));
fprintf('  n_IE = [%+.4f %+.4f %+.4f]  PC1 = %.1f%%  N = %d frames\n', ...
    n_IE, varexp_IE, size(rotVecs_IE,1));
fprintf('  Inter-axis angle = %.2f deg  [literature: 72-83 deg]\n', axesAngle_deg);

%% -------------------------------------------------------------------------
%  STEP 4 — AXIS POSITIONS
% -------------------------------------------------------------------------
q_PF = pinv(Amat_PF) * bvec_PF;
q_IE = pinv(Amat_IE) * bvec_IE;
q_PF = q_PF' - dot(q_PF', n_PF)*n_PF;
q_IE = q_IE' - dot(q_IE', n_IE)*n_IE;

%% -------------------------------------------------------------------------
%  STEP 5 — JOINT CENTRE
% -------------------------------------------------------------------------
d_skew = q_IE - q_PF;
M_skew = [ dot(n_PF,n_PF)  -dot(n_PF,n_IE);
           dot(n_PF,n_IE)  -dot(n_IE,n_IE) ];
ts     = M_skew \ [dot(d_skew,n_PF); dot(d_skew,n_IE)];
c1 = q_PF + ts(1)*n_PF;
c2 = q_IE + ts(2)*n_IE;
jointCentre = 0.5*(c1 + c2);
skewDist    = norm(c2 - c1);

fprintf('\n=== STEP 5: JOINT CENTRE (calf frame) ===\n');
fprintf('  Skew distance:  %.2f mm\n', skewDist);
fprintf('  Joint centre:   [%+.2f  %+.2f  %+.2f] mm\n', jointCentre);

%% -------------------------------------------------------------------------
%  STEP 6 — NEAREST FOOT MARKERS TO THE FOUR PLANE-CURVE INTERSECTIONS
%
%  Plane 1 (PF divider): contains [0,1,0] and n_IE → normal = cross([0,1,0], n_IE)
%  Plane 2 (IE divider): contains [0,1,0] and n_PF → normal = cross([0,1,0], n_PF)
% -------------------------------------------------------------------------
Yc_cf = [0, 1, 0];
n_plane_PF = cross(Yc_cf, n_IE);  n_plane_PF = n_plane_PF / norm(n_plane_PF);
n_plane_IE = cross(Yc_cf, n_PF);  n_plane_IE = n_plane_IE / norm(n_plane_IE);

nFootM        = numel(iFoot);
dist_plane_PF = nan(nFootM,1);
dist_plane_IE = nan(nFootM,1);

for k = 1:nFootM
    p = p0_foot_all(k,:);
    if any(isnan(p)), continue; end
    dp = p - jointCentre;
    dist_plane_PF(k) = dot(dp, n_plane_PF);
    dist_plane_IE(k) = dot(dp, n_plane_IE);
end

[~, iPF_pos] = max(dist_plane_PF);
[~, iPF_neg] = min(dist_plane_PF);
[~, iIE_pos] = max(dist_plane_IE);
[~, iIE_neg] = min(dist_plane_IE);

fprintf('\n=== STEP 6: NEAREST MARKERS TO PLANE-CURVE INTERSECTIONS ===\n');
fprintf('  +PF side (Plane1): %-10s  d=%+.1f mm\n', markerNames{iFoot(iPF_pos)}, dist_plane_PF(iPF_pos));
fprintf('  -PF side (Plane1): %-10s  d=%+.1f mm\n', markerNames{iFoot(iPF_neg)}, dist_plane_PF(iPF_neg));
fprintf('  +IE side (Plane2): %-10s  d=%+.1f mm\n', markerNames{iFoot(iIE_pos)}, dist_plane_IE(iIE_pos));
fprintf('  -IE side (Plane2): %-10s  d=%+.1f mm\n', markerNames{iFoot(iIE_neg)}, dist_plane_IE(iIE_neg));

axisResults.subject       = SUBJECT_NAME;
axisResults.method        = 'Centred PCA + Exact Angles';
axisResults.n_PF          = n_PF;
axisResults.n_IE          = n_IE;
axisResults.q_PF          = q_PF;
axisResults.q_IE          = q_IE;
axisResults.varexp_PF     = varexp_PF;
axisResults.varexp_IE     = varexp_IE;
axisResults.axesAngle_deg = axesAngle_deg;
axisResults.jointCentre   = jointCentre;
axisResults.skewDist      = skewDist;
axisResults.R_correct     = R_correct;
axisResults.bestMarkers   = struct( ...
    'PF_pos', markerNames{iFoot(iPF_pos)}, 'PF_neg', markerNames{iFoot(iPF_neg)}, ...
    'IE_pos', markerNames{iFoot(iIE_pos)}, 'IE_neg', markerNames{iFoot(iIE_neg)});

%% -------------------------------------------------------------------------
%  SAVE WORKSPACE — results struct for downstream scripts (NSGA-II MoCap)
%
%  The NSGA-II script (ankleSoftExoOpt_NSGAII_MoCap.m) loads:
%    load(ankleAxesResults_<SUBJECT>.mat, 'results')
%  and expects the following fields in results:
%
%  Identification metadata:
%    subject, method, dateComputed
%    neutralTrial, neutralFrameIdx, neutralAlignment
%
%  Neutral-frame geometry (calf frame, mm):
%    R_calf_n, O_calf_n        — calf frame rotation matrix + origin in world
%    Xc_calf, Yc_calf, Zc_calf — calf frame axis unit vectors (in world)
%    p0_foot_all               — [nFoot×3] foot marker positions
%    p0_allMarkers_inCalf      — [nMarkers×3] all neutral marker positions
%
%  Marker index arrays (into markerNames):
%    markerNames, iFoot, iCalfL, iCalfU
%
%  Rotation axes + joint centre (calf frame, mm):
%    n_PF, n_IE, jointCentre, q_PF, q_IE
%    varexp_PF, varexp_IE, axesAngle_deg, skewDist
%    R_correct
% -------------------------------------------------------------------------

results.subject            = SUBJECT_NAME;
results.method             = 'Centred PCA + Exact Angles';
results.dateComputed       = datestr(now, 'yyyy-mm-dd HH:MM:SS');

% Neutral frame identification
results.neutralTrial       = neutralTrial;
results.neutralFrameIdx    = iNeutral;
results.neutralAlignment   = bestNeutralScore;

% Calf frame at neutral (rotation matrix cols = [Xc Yc Zc], origin in world mm)
results.R_calf_n           = R_calf_n;       % [3×3]
results.O_calf_n           = O_calf_n;       % [1×3] mm, world frame
results.Xc_calf            = R_calf_n(:,1)'; % [1×3] anterior unit vector (world)
results.Yc_calf            = R_calf_n(:,2)'; % [1×3] superior unit vector (world)
results.Zc_calf            = R_calf_n(:,3)'; % [1×3] lateral unit vector (world)

% Neutral-frame marker positions (calf frame, mm)
results.p0_foot_all           = p0_foot_all;          % [nFoot×3]
results.p0_allMarkers_inCalf  = p0_allMarkers_inCalf; % [nMarkers×3]

% Marker index arrays and names
results.markerNames        = markerNames;
results.iFoot              = iFoot;
results.iCalfL             = iCalfL;
results.iCalfU             = iCalfU;

% Rotation axes + joint centre (calf frame, mm / unit vectors)
results.n_PF               = n_PF;
results.n_IE               = n_IE;
results.q_PF               = q_PF;
results.q_IE               = q_IE;
results.varexp_PF          = varexp_PF;
results.varexp_IE          = varexp_IE;
results.axesAngle_deg      = axesAngle_deg;
results.jointCentre        = jointCentre;
results.skewDist           = skewDist;
results.R_correct          = R_correct;

% Save to the path the NSGA-II script expects
matSavePath = fullfile(BASE_DIR, SUBJECT_NAME, ...
    sprintf('ankleAxesResults_%s.mat', SUBJECT_NAME));
save(matSavePath, 'results');
fprintf('\n  Workspace saved → %s\n', matSavePath);
fprintf('  Fields: %s\n', strjoin(fieldnames(results)',' | '));


%% =========================================================================
%  FIGURE 1 — Rotation vector clouds + identified axes (2x2)
% =========================================================================
figure('Name','Fig1: Rotation Vector Clouds + Axes','Color','k','Position',[30 30 1400 900]);

col_PF = [1.00 0.50 0.10];
col_IE = [0.20 0.75 1.00];

cloud_data = {rotVecs_PF, rotVecs_IE};
cloud_mu   = {mu_PF,      mu_IE     };
cloud_n    = {n_PF,       n_IE      };
cloud_sv   = {sv_PF,      sv_IE     };
cloud_ve   = {varexp_PF,  varexp_IE };
cloud_col  = {col_PF,     col_IE    };
cloud_lbl  = {'PF/DF — talocrural (FE trials)', 'Inv/Ev — subtalar (IE trials)'};
cloud_sign = {'+Z (lateral)', '+X (anterior)'};

for ci = 1:2
    rv   = cloud_data{ci};
    mu   = cloud_mu{ci};
    n    = cloud_n{ci};
    col  = cloud_col{ci};
    angs = vecnorm(rv,2,2) * 180/pi;
    mA   = max(angs)*pi/180 * 1.35;

    h = subplot(2,2,ci);
    set(h,'Color','k','XColor','w','YColor','w','ZColor','w'); hold on; grid on;
    scatter3(rv(:,1),rv(:,2),rv(:,3), 14, angs, 'filled','MarkerFaceAlpha',0.5);
    colormap(h,hot); cb=colorbar(h); cb.Color='w';
    cb.Label.String='|r| (deg)'; cb.Label.Color='w';
    plot3(mu(1)+[-mA*n(1) mA*n(1)], mu(2)+[-mA*n(2) mA*n(2)], ...
          mu(3)+[-mA*n(3) mA*n(3)], '-','Color',col,'LineWidth',4);
    scatter3(mu(1),mu(2),mu(3), 80, col,'filled','MarkerEdgeColor','w');
    xlabel('rx','Color','w'); ylabel('ry','Color','w'); zlabel('rz','Color','w');
    title(sprintf('%s\nn=[%+.3f %+.3f %+.3f]  PC1=%.1f%%  N=%d\nsign conv: %s', ...
        cloud_lbl{ci}, n, cloud_ve{ci}, size(rv,1), cloud_sign{ci}), ...
        'Color',col,'FontSize',9,'FontWeight','bold');
    view(45,25);

    h2 = subplot(2,2,ci+2);
    set(h2,'Color','k','XColor','w','YColor','w'); hold on; grid on;
    sv  = cloud_sv{ci};
    tot = sum(sv.^2);
    v3  = sv(1:3).^2/tot*100;
    barCols = {col, col*0.6+[0.3 0.3 0.3]*0.4, [0.5 0.5 0.5]};
    for pc=1:3
        bar(h2, pc, v3(pc), 0.6, 'FaceColor', barCols{pc}, 'EdgeColor','none');
        text(h2, pc, v3(pc)+1.2, sprintf('%.1f%%',v3(pc)), ...
             'Color','w','FontSize',10,'HorizontalAlignment','center','FontWeight','bold');
    end
    set(h2,'XTick',1:3,'XTickLabel',{'PC1','PC2','PC3'},'FontSize',9);
    ylim([0 110]); ylabel(h2,'Variance explained (%)','Color','w','FontSize',9);
    yline(h2, 2,'--','Color',[0.5 0.5 0.5],'LineWidth',1,'Label','~2% noise','LabelColor',[0.6 0.6 0.6]);
    title(h2, sprintf('Scree — %s', cloud_lbl{ci}),'Color','w','FontSize',9);
end
sgtitle(sprintf('%s  |  n_{PF}=[%+.3f %+.3f %+.3f]  n_{IE}=[%+.3f %+.3f %+.3f]  axes=%.1f deg', ...
    SUBJECT_NAME, n_PF, n_IE, axesAngle_deg), ...
    'Color','w','FontSize',10.5,'FontWeight','bold');

%% =========================================================================
%  FIGURE 2 — Foot marker arcs in calf frame
% =========================================================================
figure('Name','Fig2: Foot Marker Arcs','Color','k','Position',[50 700 1380 600]);
trialSetsPlot = {PF_TRIALS, IE_TRIALS};
f2Titles      = {'PF/DF trials — foot arcs in calf frame', 'Inv/Ev trials — foot arcs in calf frame'};
arcCols       = {[1 0.5 0.1], [0.3 0.8 1]};

for p = 1:2
    hAx = subplot(1,2,p);
    set(hAx,'Color','k','XColor','w','YColor','w','ZColor','w','DataAspectRatio',[1 1 1]);
    hold on; grid on;
    nTp = numel(trialSetsPlot{p});
    for t = 1:nTp
        tName = trialSetsPlot{p}{t};
        md    = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,tName), markerXCols);
        nF    = size(md,1);
        [O_c, R_c, ~] = computeCalfFrames(md,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
        traceIdx = iFoot(round(linspace(1, numel(iFoot), min(6,numel(iFoot)))));
        alpha    = 0.3 + 0.4*(t-1)/max(1, nTp-1);
        for k = 1:numel(traceIdx)
            traj = nan(nF,3);
            for f = 1:nF
                pw = squeeze(md(f, traceIdx(k), :))';
                if any(isnan(pw)), continue; end
                traj(f,:) = (R_c(:,:,f)' * (pw - O_c(f,:))')';
            end
            plot3(hAx, traj(:,1), traj(:,2), traj(:,3), ...
                  'Color',[arcCols{p}, alpha],'LineWidth',1);
        end
    end
    scatter3(hAx, p0_foot_all(:,1), p0_foot_all(:,2), p0_foot_all(:,3), ...
             40,[1 1 1],'filled','MarkerEdgeColor','k');
    drawTriad(hAx, [0 0 0], eye(3), 30, 'Calf');
    xlabel('X_{Calf} (mm)','Color','w'); ylabel('Y_{Calf} (mm)','Color','w');
    zlabel('Z_{Calf} (mm)','Color','w');
    title(f2Titles{p},'Color','w','FontSize',11); view(45,20);
end
sgtitle('Foot marker arcs in calf frame  (white = neutral)','Color','w','FontSize',13,'FontWeight','bold');

%% =========================================================================
%  FIGURE 3 — Identified axes + joint centre (3D, calf frame)
% =========================================================================
figure('Name','Fig3: Rotation Axes & Joint Centre','Color','k','Position',[500 280 900 760]);
hAx3 = axes('Color','k','XColor','w','YColor','w','ZColor','w','DataAspectRatio',[1 1 1]);
hold on; grid on;
axLen3 = 90;
pts_PF = [q_PF - axLen3*n_PF; q_PF + axLen3*n_PF];
plot3(pts_PF(:,1),pts_PF(:,2),pts_PF(:,3),'-','Color',[1 0.5 0.1],'LineWidth',3);
text(pts_PF(2,1),pts_PF(2,2),pts_PF(2,3)+4,'PF/DF axis','Color',[1 0.5 0.1],'FontSize',10,'FontWeight','bold');
pts_IE = [q_IE - axLen3*n_IE; q_IE + axLen3*n_IE];
plot3(pts_IE(:,1),pts_IE(:,2),pts_IE(:,3),'-','Color',[0.2 0.8 1],'LineWidth',3);
text(pts_IE(2,1),pts_IE(2,2),pts_IE(2,3)+4,'Inv/Ev axis','Color',[0.2 0.8 1],'FontSize',10,'FontWeight','bold');
plot3([c1(1) c2(1)],[c1(2) c2(2)],[c1(3) c2(3)],'--w','LineWidth',1.5);
scatter3(c1(1),c1(2),c1(3),80,[1 0.5 0.1],'filled','MarkerEdgeColor','w');
scatter3(c2(1),c2(2),c2(3),80,[0.2 0.8 1],'filled','MarkerEdgeColor','w');
scatter3(jointCentre(1),jointCentre(2),jointCentre(3),280,[1 0.9 0],'p','filled','MarkerEdgeColor','w','LineWidth',1.5);
text(jointCentre(1)+4,jointCentre(2)+4,jointCentre(3)+6, ...
     sprintf('Joint centre\n[%.1f, %.1f, %.1f] mm',jointCentre),'Color',[1 0.9 0],'FontSize',9,'FontWeight','bold');
drawTriad(hAx3, [0 0 0], eye(3), 45, 'Calf');
scatter3(p0_foot_all(:,1),p0_foot_all(:,2),p0_foot_all(:,3),22,[1 0.3 0.15],'filled','MarkerFaceAlpha',0.25,'MarkerEdgeColor','none');
pfMarkerCols = {[1.0 0.92 0.2], [1.0 0.55 0.1]};
pfMarkerIdx  = {iPF_pos, iPF_neg};
pfSign       = {'+n_{PF}', '-n_{PF}'};
for ki = 1:2
    p   = p0_foot_all(pfMarkerIdx{ki},:);  col = pfMarkerCols{ki};
    lbl = sprintf('%s  (%s)', markerNames{iFoot(pfMarkerIdx{ki})}, pfSign{ki});
    plot3([jointCentre(1) p(1)],[jointCentre(2) p(2)],[jointCentre(3) p(3)],'--','Color',[col 0.75],'LineWidth',1.8);
    scatter3(p(1),p(2),p(3),160,col,'filled','MarkerEdgeColor','w','LineWidth',1.2);
    text(p(1)+4,p(2)+4,p(3)+5,lbl,'Color',col,'FontSize',8,'FontWeight','bold','Interpreter','none');
end
ieMarkerCols = {[0.15 1.0 0.85], [0.05 0.65 1.0]};
ieMarkerIdx  = {iIE_pos, iIE_neg};
ieSign       = {'+n_{IE}', '-n_{IE}'};
for ki = 1:2
    p   = p0_foot_all(ieMarkerIdx{ki},:);  col = ieMarkerCols{ki};
    lbl = sprintf('%s  (%s)', markerNames{iFoot(ieMarkerIdx{ki})}, ieSign{ki});
    plot3([jointCentre(1) p(1)],[jointCentre(2) p(2)],[jointCentre(3) p(3)],'--','Color',[col 0.75],'LineWidth',1.8);
    scatter3(p(1),p(2),p(3),160,col,'filled','MarkerEdgeColor','w','LineWidth',1.2);
    text(p(1)+4,p(2)+4,p(3)+5,lbl,'Color',col,'FontSize',8,'FontWeight','bold','Interpreter','none');
end
xlabel('X_{Calf} (mm)','Color','w','FontSize',10);
ylabel('Y_{Calf} (mm)','Color','w','FontSize',10);
zlabel('Z_{Calf} (mm)','Color','w','FontSize',10);
title(sprintf('Rotation Axes & Joint Centre  |  skew dist = %.1f mm  |  axes angle = %.1f deg', ...
              skewDist, axesAngle_deg),'Color','w','FontSize',11,'FontWeight','bold');
view(45,25);

%% =========================================================================
%  FIGURE 4 — PF/DF and Inv/Ev angle time series — EXACT vs LINEAR
%
%  Solid line  = exact sequential decomposition (Grood-Suntay / JCS style)
%  Dashed line = linear projection / BCH approximation (original method)
%  Difference panel shows (exact - linear) to quantify BCH error magnitude.
% =========================================================================
allTrials = [PF_TRIALS, IE_TRIALS];
nT        = numel(allTrials);
tCols4    = {[1 0.5 0.1],[1 0.75 0.2],[0.25 0.8 1],[0.1 0.5 0.9]};

% 3 columns: PF/DF | Inv/Ev | Difference (exact - linear)
figure('Name','Fig4: Exact vs Linear Angle Time Series','Color','k', ...
       'Position',[60 60 1700 min(900, 210*nT)]);

for t = 1:nT
    tName = allTrials{t};
    md    = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,tName), markerXCols);
    nF    = size(md,1);
    time  = (0:nF-1) / 100;

    [~, R_c, calfV] = computeCalfFrames(md,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
    [~, R_f, ~]     = computeFootFrames(md, iFoot, calfV);

    % Preallocate
    ang_PF_ex  = nan(nF,1);   ang_IE_ex  = nan(nF,1);   % exact
    ang_PF_lin = nan(nF,1);   ang_IE_lin = nan(nF,1);   % linear (BCH)

    for f = 1:nF
        R_rel = R_c(:,:,f)' * R_f(:,:,f) * R_correct;
        rv    = rotmat2vec(R_rel);
        if isempty(rv) || all(rv==0), continue; end

        % --- EXACT decomposition ---
        [ang_PF_ex(f), ang_IE_ex(f)] = decomposeAnglesExact(R_rel, n_PF, n_IE);

        % --- LINEAR projection (BCH) for comparison ---
        ang_PF_lin(f) = dot(rv, n_PF) * 180/pi;
        ang_IE_lin(f) = dot(rv, n_IE) * 180/pi;
    end

    col = tCols4{mod(t-1,numel(tCols4))+1};

    % --- PF/DF column ---
    ax1 = subplot(nT,3,3*t-2);
    plot(time, ang_PF_ex, '-', 'Color',col,'LineWidth',2.0); hold on;
    plot(time, ang_PF_lin,'--','Color',col*0.65+[0.25 0.25 0.25],'LineWidth',1.0);
    yline(0,'--','Color',[1 1 1 0.3]);
    ylabel('PF/DF (deg)','Color','w'); xlabel('Time (s)','Color','w');
    title(sprintf('%s — PF/DF  (solid=exact, dash=linear)',tName), ...
          'Color','w','FontSize',8,'Interpreter','none');
    set(ax1,'Color','k','XColor','w','YColor','w');

    % --- Inv/Ev column ---
    ax2 = subplot(nT,3,3*t-1);
    plot(time, ang_IE_ex, '-', 'Color',col,'LineWidth',2.0); hold on;
    plot(time, ang_IE_lin,'--','Color',col*0.65+[0.25 0.25 0.25],'LineWidth',1.0);
    yline(0,'--','Color',[1 1 1 0.3]);
    ylabel('Inv/Ev (deg)','Color','w'); xlabel('Time (s)','Color','w');
    title(sprintf('%s — Inv/Ev  (solid=exact, dash=linear)',tName), ...
          'Color','w','FontSize',8,'Interpreter','none');
    set(ax2,'Color','k','XColor','w','YColor','w');

    % --- Difference column (exact - linear) ---
    ax3 = subplot(nT,3,3*t);
    diff_PF = ang_PF_ex - ang_PF_lin;
    diff_IE = ang_IE_ex - ang_IE_lin;
    plot(time, diff_PF, '-', 'Color',[1 0.5 0.1],'LineWidth',1.5,'DisplayName','PF diff'); hold on;
    plot(time, diff_IE, '-', 'Color',[0.2 0.75 1],'LineWidth',1.5,'DisplayName','IE diff');
    yline(0,'--','Color',[1 1 1 0.3]);
    ylabel('\Delta\theta (deg)','Color','w'); xlabel('Time (s)','Color','w');
    rms_pf = sqrt(nanmean(diff_PF.^2));
    rms_ie = sqrt(nanmean(diff_IE.^2));
    title(sprintf('%s — Exact minus Linear  |  RMS: PF=%.1f IE=%.1f deg', ...
          tName, rms_pf, rms_ie),'Color','w','FontSize',8,'Interpreter','none');
    set(ax3,'Color','k','XColor','w','YColor','w');
    legend(ax3,'show','TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'FontSize',7,'Location','best');
end
sgtitle(sprintf('%s  —  Exact angle decomposition vs linear projection (BCH)', SUBJECT_NAME), ...
        'Color','w','FontSize',12,'FontWeight','bold');

%% =========================================================================
%  FIGURE 5 — LIVE ANIMATION (uses EXACT angles in readout)
% =========================================================================
if SHOW_ANIMATION

fprintf('\n--- Pre-computing visualisation for trial: %s ---\n', VIS_TRIAL);
mdV = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,VIS_TRIAL), markerXCols);
nFV = size(mdV,1);

[O_cV, R_cV, calfVV] = computeCalfFrames(mdV,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
[~,    R_fV, ~]      = computeFootFrames(mdV, iFoot, calfVV);

dispDataV  = nan(nFV, nMarkers, 3);
footTriadR = zeros(3, 3, nFV);
ang_PF_V   = zeros(nFV,1);
ang_IE_V   = zeros(nFV,1);

for f = 1:nFV
    Rc = R_cV(:,:,f);  Oc = O_cV(f,:);
    for mv = 1:nMarkers
        pv = squeeze(mdV(f,mv,:))';
        if ~any(isnan(pv))
            dispDataV(f,mv,:) = (Rc' * (pv - Oc)')';
        end
    end
    footTriadR(:,:,f) = Rc' * R_fV(:,:,f) * R_correct;
    % EXACT angles for live readout
    [aPF, aIE] = decomposeAnglesExact(footTriadR(:,:,f), n_PF, n_IE);
    if ~isnan(aPF), ang_PF_V(f) = aPF; end
    if ~isnan(aIE), ang_IE_V(f) = aIE; end
end

allPtsV = reshape(dispDataV, [], 3);
validV  = allPtsV(~any(isnan(allPtsV),2), :);
if ~isempty(validV)
    ptMin = min(validV) - 25;  ptMax = max(validV) + 25;
else
    ptMin = [-200 -200 -200];  ptMax = [200 200 200];
end

hFig5 = figure('Name', sprintf('Fig5: %s — %s  [calf frame + exact angles]', SUBJECT_NAME, VIS_TRIAL), ...
               'Color','k','Position',[100 100 1100 820]);
hAx5  = axes('Parent',hFig5,'Color','k','XColor','w','YColor','w','ZColor','w', ...
             'DataAspectRatio',[1 1 1], ...
             'XLim',[ptMin(1) ptMax(1)],'YLim',[ptMin(2) ptMax(2)],'ZLim',[ptMin(3) ptMax(3)]);
hold(hAx5,'on'); grid(hAx5,'on');
xlabel(hAx5,'X_{Calf} (mm)','Color','w');
ylabel(hAx5,'Y_{Calf} (mm)','Color','w');
zlabel(hAx5,'Z_{Calf} (mm)','Color','w');
view(hAx5, 0, 90);

frameList = 1:VIS_FRAME_SKIP:nFV;
if strcmp(VIS_MODE,'frame')
    fIdx = max(1, min(VIS_FRAME, nFV));
    renderAxisFrame(hAx5, dispDataV, markerNames, markerColours, ...
        fIdx, nFV, SUBJECT_NAME, VIS_TRIAL, ...
        n_PF, q_PF, n_IE, q_IE, jointCentre, ...
        footTriadR, ang_PF_V, ang_IE_V, ...
        AXIS_SCALE, SHOW_LABELS, ...
        iPF_pos, iPF_neg, iIE_pos, iIE_neg, iFoot, p0_foot_all);
else
    dt = (VIS_FRAME_SKIP/100) / VIS_SPEED;
    for fi = 1:numel(frameList)
        if ~ishandle(hFig5), break; end
        fIdx = frameList(fi);  t0 = tic;
        renderAxisFrame(hAx5, dispDataV, markerNames, markerColours, ...
            fIdx, nFV, SUBJECT_NAME, VIS_TRIAL, ...
            n_PF, q_PF, n_IE, q_IE, jointCentre, ...
            footTriadR, ang_PF_V, ang_IE_V, ...
            AXIS_SCALE, SHOW_LABELS, ...
            iPF_pos, iPF_neg, iIE_pos, iIE_neg, iFoot, p0_foot_all);
        pause(max(0, dt - toc(t0)));
    end
end
end  % SHOW_ANIMATION

%% =========================================================================
%  FIGURE 6 — Joint angle space diagnostic (EXACT angles)
% =========================================================================
DIAG_TRIALS = {'FE01','FE02','IE01','IE02', ...
               'FEIE01','FEIE02','IEFE01','IEFE02', ...
               'Random01','Random02','Random03'};
TRIAL_BCOL = {[1 0.5 0.1],[1 0.5 0.1],[0.2 0.75 1],[0.2 0.75 1], ...
              [0.8 0.5 1],[0.8 0.5 1],[0.8 0.5 1],[0.8 0.5 1], ...
              [0.2 0.9 0.4],[0.2 0.9 0.4],[0.2 0.9 0.4]};

fprintf('\n--- Computing Figure 6: joint angle space (exact decomposition) ---\n');

nDT   = numel(DIAG_TRIALS);
allPF = [];  allIE = [];  allResid = [];

data_PF    = cell(nDT,1);
data_IE    = cell(nDT,1);
data_resid = cell(nDT,1);
data_time  = cell(nDT,1);

for ti = 1:nDT
    tName = DIAG_TRIALS{ti};
    md    = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,tName), markerXCols);
    nF    = size(md,1);
    [~, R_c, calfV] = computeCalfFrames(md,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
    [~, R_f, ~]     = computeFootFrames(md, iFoot, calfV);

    tPF = nan(nF,1);  tIE = nan(nF,1);  tRes = nan(nF,1);
    for f = 1:nF
        R_rel_f = R_c(:,:,f)' * R_f(:,:,f) * R_correct;
        rv      = rotmat2vec(R_rel_f);
        if isempty(rv) || norm(rv)*180/pi < MIN_ANGLE_DEG, continue; end

        % EXACT angles
        [tPF(f), tIE(f)] = decomposeAnglesExact(R_rel_f, n_PF, n_IE);

        % Residual: rotation remaining after removing both exact components
        %   R_resid = R(n_PF,-tPF) * R(n_IE,-tIE) * R_rel  (should = I for 2-DoF)
        R_resid  = rodrigues(n_PF, -tPF(f)*pi/180) * rodrigues(n_IE, -tIE(f)*pi/180) * R_rel_f;
        rv_r     = rotmat2vec(R_resid);
        if isempty(rv_r), tRes(f) = 0;
        else,             tRes(f) = norm(rv_r) * 180/pi; end
    end
    valid = ~isnan(tPF);
    data_PF{ti}    = tPF(valid);
    data_IE{ti}    = tIE(valid);
    data_resid{ti} = tRes(valid);
    data_time{ti}  = find(valid) / 100;

    allPF    = [allPF;    data_PF{ti}];    %#ok<AGROW>
    allIE    = [allIE;    data_IE{ti}];    %#ok<AGROW>
    allResid = [allResid; data_resid{ti}]; %#ok<AGROW>
end

lim99 = prctile(max(abs(allPF), abs(allIE)), 99) * 1.12;
lim99 = ceil(lim99 / 5) * 5;

figure('Name','Fig6: Joint Angle Space (Exact)','Color','k','Position',[30 30 1760 1080]);
colmap = [linspace(0.2,1,256)', linspace(0.4,0.1,256)', linspace(0.9,0.1,256)'];

for ti = 1:nDT
    h = subplot(3,4,ti);
    set(h,'Color',[0.04 0.04 0.04],'XColor','w','YColor','w'); hold on; grid on;
    set(h,'GridColor',[0.3 0.3 0.3],'GridAlpha',0.5);

    pf = data_PF{ti};  ie = data_IE{ti};  t  = data_time{ti};
    N  = numel(pf);
    if N < 3
        title(h, sprintf('%s  (no data)', DIAG_TRIALS{ti}), 'Color','w','FontSize',9); continue;
    end

    tcol = (t - min(t)) / max(t - min(t) + eps);
    cidx = max(1, min(256, round(tcol * 255) + 1));
    cols = colmap(cidx,:);
    for k = 1:N
        plot(h, pf(k), ie(k), '.', 'Color', cols(k,:), 'MarkerSize', 4);
    end
    xline(h,0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.6,'Alpha',0.7);
    yline(h,0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.6,'Alpha',0.7);

    rho   = corr(pf, ie);
    rms_e = sqrt(mean(data_resid{ti}.^2));
    var_pf = var(pf);  var_ie = var(ie);  var_res = var(data_resid{ti});
    v2d   = (var_pf + var_ie) / (var_pf + var_ie + var_res) * 100;
    bc = TRIAL_BCOL{ti};
    set(h,'XColor',bc,'YColor',bc);
    xlim(h,[-lim99 lim99]);  ylim(h,[-lim99 lim99]);
    xlabel(h,'\theta_{PF} (deg)','Color','w','FontSize',8);
    ylabel(h,'\theta_{IE} (deg)','Color','w','FontSize',8);
    title(h, sprintf('%s  |  r=%+.3f  RMS_e=%.1f deg  V2D=%.0f%%  N=%d', ...
          DIAG_TRIALS{ti}, rho, rms_e, v2d, N), ...
          'Color',bc,'FontSize',8,'FontWeight','bold','Interpreter','none');
    set(h,'XTick',-lim99:20:lim99,'YTick',-lim99:20:lim99,'FontSize',7);
end

h12 = subplot(3,4,12);
set(h12,'Color',[0.04 0.04 0.04],'XColor','w','YColor','w'); hold on; grid on;
groupBounds = {1:2, 3:4, 5:8, 9:11};
groupCols   = {[1 0.5 0.1],[0.2 0.75 1],[0.8 0.5 1],[0.2 0.9 0.4]};
groupNames  = {'FE (training)','IE (training)','FEIE/IEFE','Random'};
for gi = 1:4
    for ti = groupBounds{gi}
        plot(h12, data_PF{ti}, data_IE{ti}, '.', 'Color', [groupCols{gi} 0.35], 'MarkerSize', 3);
    end
end
xline(h12,0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.6);
yline(h12,0,'--','Color',[0.5 0.5 0.5],'LineWidth',0.6);
xlim(h12,[-lim99 lim99]);  ylim(h12,[-lim99 lim99]);
xlabel(h12,'\theta_{PF} (deg)','Color','w','FontSize',8);
ylabel(h12,'\theta_{IE} (deg)','Color','w','FontSize',8);
for gi = 1:4
    plot(h12, nan, nan, '.', 'Color', groupCols{gi}, 'MarkerSize', 10, 'DisplayName', groupNames{gi});
end
legend(h12,'show','TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'FontSize',7,'Location','southeast');

rho_pool  = corr(allPF, allIE);
rms_pool  = sqrt(mean(allResid.^2));
var_all   = var(allPF)+var(allIE); var_all_r = var(allResid);
v2d_pool  = var_all/(var_all+var_all_r)*100;
title(h12, sprintf('Pooled  |  r=%+.3f  RMS_e=%.1f deg  V2D=%.0f%%', rho_pool, rms_pool, v2d_pool), ...
      'Color','w','FontSize',8,'FontWeight','bold','Interpreter','none');
set(h12,'XTick',-lim99:20:lim99,'YTick',-lim99:20:lim99,'FontSize',7);
cb = colorbar(h12); colormap(h12, colmap);
cb.Label.String='Time (blue=start, red=end)'; cb.Color='w'; cb.Label.Color='w'; cb.FontSize=7;

sgtitle(sprintf('%s — Joint angle space (EXACT decomp.)  |  pooled r=%.3f  V2D=%.0f%%', ...
    SUBJECT_NAME, rho_pool, v2d_pool), 'Color','w','FontSize',10,'FontWeight','bold');

fprintf('Pooled stats (EXACT angles):\n');
fprintf('  r(theta_PF, theta_IE) = %+.4f\n', rho_pool);
fprintf('  RMS residual          = %.2f deg\n', rms_pool);
fprintf('  Var2D                 = %.1f%%\n', v2d_pool);

%% =========================================================================
%  FIGURE 7 — DoF dimensionality test (rotation vector residual unchanged —
%  PC analysis is on raw rotation vectors, not the decomposed angles)
% =========================================================================
fprintf('\n--- Computing Figure 7: DoF dimensionality test ---\n');

DOF_TRIAL_GROUPS = { ...
    {'FE01','FE02'},             'FE training',  [1.00 0.55 0.10]; ...
    {'IE01','IE02'},             'IE training',  [0.20 0.75 1.00]; ...
    {'FEIE01','FEIE02','IEFE01','IEFE02'}, 'Combined', [0.80 0.50 1.00]; ...
    {'Random01','Random02','Random03'},    'Random',   [0.20 0.90 0.40]};

nGroups = size(DOF_TRIAL_GROUPS,1);
group_scree   = zeros(nGroups, 3);
group_pc3dirs = zeros(nGroups, 3);

rv_groups = cell(nGroups, 1);
for gi = 1:nGroups
    trialList = DOF_TRIAL_GROUPS{gi,1};
    rv_g = [];
    for ti = 1:numel(trialList)
        tName = trialList{ti};
        md    = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,tName), markerXCols);
        nF    = size(md,1);
        [~, R_c, calfV] = computeCalfFrames(md,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
        [~, R_f, ~]     = computeFootFrames(md, iFoot, calfV);
        for f = 1:nF
            rv = rotmat2vec(R_c(:,:,f)' * R_f(:,:,f) * R_correct);
            if isempty(rv), continue; end
            ang = norm(rv)*180/pi;
            if ang < MIN_ANGLE_DEG || ang > MAX_ANGLE_DEG_POS, continue; end
            rv_g(end+1,:) = rv; %#ok<SAGROW>
        end
    end
    rv_groups{gi} = rv_g;
    if size(rv_g,1) > 3
        mu_g = mean(rv_g,1);
        sv_g = svd(rv_g - mu_g, 'econ');
        tot  = sum(sv_g.^2);
        group_scree(gi,:) = sv_g(1:3).^2 / tot * 100;
        [~,~,V_g] = svd(rv_g - mu_g, 'econ');
        group_pc3dirs(gi,:) = V_g(:,3)';
    end
    fprintf('  %s: PC1=%.1f%%  PC2=%.1f%%  PC3=%.1f%%\n', ...
        DOF_TRIAL_GROUPS{gi,2}, group_scree(gi,1), group_scree(gi,2), group_scree(gi,3));
end

rand_t = {}; rand_pc3mag = {}; rand_resmag = {}; rand_totmag = {};
rand_rv_pool = [];
for ti = 1:3
    tName = sprintf('Random0%d',ti);
    md    = loadTrialData(csvPathFor(BASE_DIR,SUBJECT_NAME,tName), markerXCols);
    nF    = size(md,1);
    [~, R_c, calfV] = computeCalfFrames(md,iAnkle3,iAnkle5,iKnee1,iKnee2,iCalfU,iCalfL);
    [~, R_f, ~]     = computeFootFrames(md, iFoot, calfV);
    pc3_rand = group_pc3dirs(4,:);
    tSer=[]; pc3Ser=[]; resSer=[]; totSer=[];
    for f = 1:nF
        R_rel_f = R_c(:,:,f)' * R_f(:,:,f) * R_correct;
        rv = rotmat2vec(R_rel_f);
        if isempty(rv), continue; end
        ang = norm(rv)*180/pi;
        if ang < MIN_ANGLE_DEG, continue; end
        % Residual using EXACT decomposition
        [aPF, aIE] = decomposeAnglesExact(R_rel_f, n_PF, n_IE);
        if isnan(aPF) || isnan(aIE)
            rv_r_mag = 0;
        else
            R_resid  = rodrigues(n_PF, -aPF*pi/180) * rodrigues(n_IE, -aIE*pi/180) * R_rel_f;
            rv_r     = rotmat2vec(R_resid);
            rv_r_mag = norm(rv_r) * 180/pi;
        end
        tSer(end+1)   = (f-1)/100; %#ok<SAGROW>
        pc3Ser(end+1) = abs(dot(rv, pc3_rand)) * 180/pi; %#ok<SAGROW>
        resSer(end+1) = rv_r_mag; %#ok<SAGROW>
        totSer(end+1) = ang; %#ok<SAGROW>
        rand_rv_pool(end+1,:) = rv; %#ok<SAGROW>
    end
    rand_t{ti}=tSer; rand_pc3mag{ti}=pc3Ser; rand_resmag{ti}=resSer; rand_totmag{ti}=totSer;
end

figure('Name','Fig7: DoF Dimensionality Test (Exact residual)','Color','k','Position',[40 40 1600 900]);

h1 = subplot(2,2,1);
set(h1,'Color','k','XColor','w','YColor','w'); hold on; grid on;
barW = 0.22;
barCols  = {[0.4 0.9 0.4],[1.0 0.85 0.2],[0.9 0.3 0.3]};
pcLabels = {'PC1','PC2','PC3'};
offsets  = [-barW, 0, barW];
for pc = 1:3
    for gi = 1:nGroups
        bar(h1, gi+offsets(pc), group_scree(gi,pc), barW*0.9, 'FaceColor',barCols{pc},'EdgeColor','none');
        text(h1, gi+offsets(pc), group_scree(gi,pc)+0.8, sprintf('%.1f',group_scree(gi,pc)), ...
             'Color','w','FontSize',7,'HorizontalAlignment','center');
    end
end
for pc=1:3, bar(h1,nan,nan,'FaceColor',barCols{pc},'EdgeColor','none','DisplayName',pcLabels{pc}); end
legend(h1,'show','TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'FontSize',8);
set(h1,'XTick',1:nGroups,'XTickLabel',{DOF_TRIAL_GROUPS{:,2}},'FontSize',8);
ylim(h1,[0 105]); ylabel(h1,'Variance explained (%)','Color','w','FontSize',9);
title(h1,'Scree per trial group','Color','w','FontSize',9,'FontWeight','bold');
yline(h1, 2,'--','Color',[0.6 0.6 0.6],'LineWidth',1,'Label','noise floor ~2%','LabelColor',[0.6 0.6 0.6],'FontSize',7);

h2 = subplot(2,2,2);
set(h2,'Color','k','XColor','w','YColor','w'); hold on; grid on;
rndCols = {[0.3 0.9 0.4],[0.3 0.7 0.9],[0.9 0.7 0.3]};
for ti = 1:3
    plot(h2,rand_t{ti},rand_pc3mag{ti},'-','Color',[rndCols{ti} 0.7],'LineWidth',1,'DisplayName',sprintf('Random0%d',ti));
    if numel(rand_pc3mag{ti})>20
        plot(h2,rand_t{ti},movmean(rand_pc3mag{ti},30),'-','Color',rndCols{ti},'LineWidth',2.5);
    end
end
legend(h2,'show','TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'FontSize',8,'Location','northeast');
for ti=1:3, plot(h2,rand_t{ti},rand_resmag{ti},'--','Color',[rndCols{ti} 0.4],'LineWidth',0.8); end
text(h2,0.02,0.96,'Solid=PC3 proj.  Dashed=exact residual','Units','normalized','Color',[0.75 0.75 0.75],'FontSize',7);
xlabel(h2,'Time (s)','Color','w','FontSize',9); ylabel(h2,'Angle (deg)','Color','w','FontSize',9);
title(h2,'PC3 projection vs time — flat=offset, varying=DoF','Color','w','FontSize',9,'FontWeight','bold');

h3 = subplot(2,2,3);
set(h3,'Color','k','XColor','w','YColor','w'); hold on; grid on;
for ti=1:3
    scatter(h3,rand_totmag{ti},rand_resmag{ti},8,rndCols{ti},'filled','MarkerFaceAlpha',0.3,'DisplayName',sprintf('Random0%d',ti));
end
all_tot=[rand_totmag{:}]; all_res=[rand_resmag{:}];
p_fit=polyfit(all_tot,all_res,1);
xfit=linspace(0,max(all_tot),100); yfit=polyval(p_fit,xfit);
plot(h3,xfit,yfit,'-w','LineWidth',2,'DisplayName',sprintf('fit slope=%.3f',p_fit(1)));
legend(h3,'show','TextColor','w','Color','none','EdgeColor',[0.4 0.4 0.4],'FontSize',7,'Location','northwest');
xlabel(h3,'Total rotation |r| (deg)','Color','w','FontSize',9);
ylabel(h3,'Exact residual (deg)','Color','w','FontSize',9);
title(h3,'Exact residual vs total angle — slope>0 = BCH, flat = offset','Color','w','FontSize',9,'FontWeight','bold');
r_res_tot = corr(all_tot',all_res');
text(h3,0.98,0.05,sprintf('r(res,|r|) = %.3f',r_res_tot),'Units','normalized','Color','w','FontSize',9,'HorizontalAlignment','right','FontWeight','bold');

h4 = subplot(2,2,4);
set(h4,'Color','k','XColor','w','YColor','w','ZColor','w'); hold on; grid on;
set(h4,'DataAspectRatio',[1 1 1]);
if ~isempty(rand_rv_pool)
    pc3_rand = group_pc3dirs(4,:);
    pc3_proj = rand_rv_pool * pc3_rand' * 180/pi;
    lim_c    = prctile(abs(pc3_proj), 98);
    cdat     = pc3_proj / lim_c;
    cm = [linspace(0.2,1,128)', linspace(0.4,1,128)', linspace(0.9,1,128)'; ...
          linspace(1,0.9,128)', linspace(1,0.3,128)', linspace(1,0.2,128)'];
    cidx = max(1, min(256, round((cdat+1)/2 * 255) + 1));
    cols_4 = cm(cidx,:);
    for k = 1:size(rand_rv_pool,1)
        plot3(h4, rand_rv_pool(k,1), rand_rv_pool(k,2), rand_rv_pool(k,3), '.','Color',cols_4(k,:),'MarkerSize',3);
    end
    pc3_sc = max(vecnorm(rand_rv_pool,2,2)) * 1.2;
    plot3(h4,[-pc3_sc*pc3_rand(1) pc3_sc*pc3_rand(1)],[-pc3_sc*pc3_rand(2) pc3_sc*pc3_rand(2)],[-pc3_sc*pc3_rand(3) pc3_sc*pc3_rand(3)],'-','Color',[0.9 0.3 0.3],'LineWidth',3);
    plot3(h4,[-pc3_sc*n_PF(1) pc3_sc*n_PF(1)],[-pc3_sc*n_PF(2) pc3_sc*n_PF(2)],[-pc3_sc*n_PF(3) pc3_sc*n_PF(3)],'-','Color',[1 0.5 0.1],'LineWidth',2.5);
    plot3(h4,[-pc3_sc*n_IE(1) pc3_sc*n_IE(1)],[-pc3_sc*n_IE(2) pc3_sc*n_IE(2)],[-pc3_sc*n_IE(3) pc3_sc*n_IE(3)],'-','Color',[0.2 0.75 1],'LineWidth',2.5);
end
xlabel(h4,'rx','Color','w','FontSize',8); ylabel(h4,'ry','Color','w','FontSize',8); zlabel(h4,'rz','Color','w','FontSize',8);
title(h4,'Rot-vec cloud coloured by PC3  |  PC3(red)  n_{PF}(orange)  n_{IE}(cyan)','Color','w','FontSize',9,'FontWeight','bold');
view(h4, 35, 20);

sgtitle(sprintf('%s — DoF test (exact residual)  |  Random PC3=%.1f%%  |  r(res,|r|)=%.3f  BCH slope=%.4f', ...
    SUBJECT_NAME, group_scree(4,3), r_res_tot, p_fit(1)), 'Color','w','FontSize',10,'FontWeight','bold');

%% =========================================================================
%  STEP 7 — COMPLEMENTARY PLANE SETS
%
%  After identifying the rotation axes n_PF and n_IE (calf frame), two
%  complementary sets of planes are defined through the joint centre:
%
%  SET A — plane NORMAL equals a rotation axis
%    A1: normal = n_PF  →  perpendicular to the PF/DF axis
%    A2: normal = n_IE  →  perpendicular to the Inv/Ev axis
%
%  SET B — plane CONTAINS a rotation axis AND the calf-superior axis (CalfY)
%    B1: contains {n_PF, CalfY}  →  normal = cross(n_PF, CalfY)
%    B2: contains {n_IE, CalfY}  →  normal = cross(n_IE, CalfY)
%  Note: nB1 = -n_plane_IE, nB2 = -n_plane_PF from Step 6 (same planes,
%        just opposite normal-sign convention).
% =========================================================================

CalfY_cf_p = [0, 1, 0];   % ISB calf-superior unit axis in calf frame

% --- Set A normals (= rotation axes themselves) ---------------------------
nA1 = n_PF / norm(n_PF);
nA2 = n_IE / norm(n_IE);

% --- Set B normals (cross of axis with CalfY) -----------------------------
nB1_raw = cross(n_PF, CalfY_cf_p);
nB2_raw = cross(n_IE, CalfY_cf_p);
if norm(nB1_raw) < 1e-6
    warning('n_PF is parallel to CalfY — Set B Plane B1 is degenerate.');
    nB1 = [1 0 0];
else
    nB1 = nB1_raw / norm(nB1_raw);
end
if norm(nB2_raw) < 1e-6
    warning('n_IE is parallel to CalfY — Set B Plane B2 is degenerate.');
    nB2 = [0 0 1];
else
    nB2 = nB2_raw / norm(nB2_raw);
end

fprintf('\n=== STEP 7: COMPLEMENTARY PLANE SETS (calf frame) ===\n');
fprintf('  SET A — plane normal = rotation axis:\n');
fprintf('    A1: n=[%+.4f %+.4f %+.4f]  (perp to PF/DF axis)\n', nA1);
fprintf('    A2: n=[%+.4f %+.4f %+.4f]  (perp to Inv/Ev axis)\n', nA2);
fprintf('  SET B — plane contains axis + CalfY:\n');
fprintf('    B1: n=[%+.4f %+.4f %+.4f]  (contains n_PF and CalfY)\n', nB1);
fprintf('    B2: n=[%+.4f %+.4f %+.4f]  (contains n_IE and CalfY)\n', nB2);
fprintf('  Orthogonality checks:\n');
fprintf('    dot(n_PF, nB1) = %.2e  (should be ~0)\n', dot(n_PF, nB1));
fprintf('    dot(n_IE, nB2) = %.2e  (should be ~0)\n', dot(n_IE, nB2));
fprintf('    dot(CalfY, nB1) = %.2e  (should be ~0)\n', dot(CalfY_cf_p, nB1));
fprintf('    dot(CalfY, nB2) = %.2e  (should be ~0)\n', dot(CalfY_cf_p, nB2));

% Gather neutral-frame marker positions (already in calf frame)
validFoot_plan  = p0_foot_all(all(~isnan(p0_foot_all), 2), :);
pCalfL_all_raw  = p0_allMarkers_inCalf(iCalfL, :);
pCalfL_cf_plan  = pCalfL_all_raw(all(~isnan(pCalfL_all_raw), 2), :);

% Valid marker names (parallel to validFoot_plan and pCalfL_cf_plan)
vFoot_mask   = all(~isnan(p0_foot_all), 2);
vCalfL_mask  = all(~isnan(pCalfL_all_raw), 2);
vFoot_names  = markerNames(iFoot(vFoot_mask));
vCalfL_names = markerNames(iCalfL(vCalfL_mask));

% Dynamic plane half-size — cover all markers + 30% margin
jcXZ_d  = [jointCentre(1), jointCentre(3)];
allXZ_d = [validFoot_plan(:,[1 3]); pCalfL_cf_plan(:,[1 3])];
planeHS = max(sqrt(sum((allXZ_d - jcXZ_d).^2, 2))) * 1.30;
planeHS = max(planeHS, 180);           % minimum 180 mm

% Visual parameters
axLen_plan  = 80;                      % mm — axis arrow length
col_PF_p    = [0.85  0.25  0.02];     % dark orange-red  — PF/DF
col_IE_p    = [0.05  0.45  0.80];     % medium blue      — Inv/Ev
col_CalfY_p = [0.00  0.55  0.00];     % dark green       — CalfY reference

%% =========================================================================
%  STEP 8 — ACTUATOR CANDIDATE MARKERS (Set B, XZ proximity)
%
%  For each DoF we seek insertions lying close to the plane that CONTAINS
%  the OTHER axis — those insertions produce minimal moment about that axis
%  and maximal moment about the target axis:
%
%    Near B2 (contains n_IE + CalfY) → minimal IE moment arm → PF/DF pair
%    Near B1 (contains n_PF + CalfY) → minimal PF moment arm → IE/EV pair
%
%  Within each target-plane set, the two candidates on OPPOSITE sides of
%  the partner plane form an antagonist pair (opposite moment arm signs).
%
%  4 actuator candidates:
%    PF+  nearest to B2,  on +B1 side   (Foot + CalfL insertion)
%    PF-  nearest to B2,  on -B1 side   (Foot + CalfL insertion) ← antagonist to PF+
%    IE+  nearest to B1,  on +B2 side   (Foot + CalfL insertion)
%    IE-  nearest to B1,  on -B2 side   (Foot + CalfL insertion) ← antagonist to IE+
% =========================================================================

% XZ-unit normals for Set B
nB1_xz  = [nB1(1), nB1(3)];  nB1_xz = nB1_xz / max(norm(nB1_xz), 1e-10);
nB2_xz  = [nB2(1), nB2(3)];  nB2_xz = nB2_xz / max(norm(nB2_xz), 1e-10);

% XZ-unit normals for Set A (for Fig 8 labels)
nA1_xz  = [nA1(1), nA1(3)];  nA1_xz = nA1_xz / max(norm(nA1_xz), 1e-10);
nA2_xz  = [nA2(1), nA2(3)];  nA2_xz = nA2_xz / max(norm(nA2_xz), 1e-10);

% Signed XZ distances from each plane trace
foot_XZ   = validFoot_plan(:,[1 3]);
calfL_XZ  = pCalfL_cf_plan(:,[1 3]);

dB1_foot  = (foot_XZ  - jcXZ_d) * nB1_xz';
dB2_foot  = (foot_XZ  - jcXZ_d) * nB2_xz';
dB1_calfL = (calfL_XZ - jcXZ_d) * nB1_xz';
dB2_calfL = (calfL_XZ - jcXZ_d) * nB2_xz';

dA1_foot  = (foot_XZ  - jcXZ_d) * nA1_xz';
dA2_foot  = (foot_XZ  - jcXZ_d) * nA2_xz';
dA1_calfL = (calfL_XZ - jcXZ_d) * nA1_xz';
dA2_calfL = (calfL_XZ - jcXZ_d) * nA2_xz';

% --- Find candidates (Set B) ----------------------------------------------
[~, iPFp_foot]  = findClosestOnSide(dB2_foot,  dB1_foot,  +1);  % PF+: near B2, +B1
[~, iPFn_foot]  = findClosestOnSide(dB2_foot,  dB1_foot,  -1);  % PF-: near B2, -B1
[~, iPFp_calf]  = findClosestOnSide(dB2_calfL, dB1_calfL, +1);
[~, iPFn_calf]  = findClosestOnSide(dB2_calfL, dB1_calfL, -1);
[~, iIEp_foot]  = findClosestOnSide(dB1_foot,  dB2_foot,  +1);  % IE+: near B1, +B2
[~, iIEn_foot]  = findClosestOnSide(dB1_foot,  dB2_foot,  -1);  % IE-: near B1, -B2
[~, iIEp_calf]  = findClosestOnSide(dB1_calfL, dB2_calfL, +1);
[~, iIEn_calf]  = findClosestOnSide(dB1_calfL, dB2_calfL, -1);

% --- Find candidates (Set A) — used for Fig 8 labels ---------------------
[~, iA_PFp_foot]  = findClosestOnSide(dA2_foot,  dA1_foot,  +1);
[~, iA_PFn_foot]  = findClosestOnSide(dA2_foot,  dA1_foot,  -1);
[~, iA_PFp_calf]  = findClosestOnSide(dA2_calfL, dA1_calfL, +1);
[~, iA_PFn_calf]  = findClosestOnSide(dA2_calfL, dA1_calfL, -1);
[~, iA_IEp_foot]  = findClosestOnSide(dA1_foot,  dA2_foot,  +1);
[~, iA_IEn_foot]  = findClosestOnSide(dA1_foot,  dA2_foot,  -1);
[~, iA_IEp_calf]  = findClosestOnSide(dA1_calfL, dA2_calfL, +1);
[~, iA_IEn_calf]  = findClosestOnSide(dA1_calfL, dA2_calfL, -1);

% --- Report ---------------------------------------------------------------
act_roles = {'PF+','PF-','IE+','IE-'};

% Set B tables
act_fi_b  = {iPFp_foot,   iPFn_foot,   iIEp_foot,   iIEn_foot  };
act_ci_b  = {iPFp_calf,   iPFn_calf,   iIEp_calf,   iIEn_calf  };
act_dtF_b = {dB2_foot,    dB2_foot,    dB1_foot,    dB1_foot   };
act_dtC_b = {dB2_calfL,   dB2_calfL,   dB1_calfL,   dB1_calfL  };
act_dpF_b = {dB1_foot,    dB1_foot,    dB2_foot,    dB2_foot   };
act_dpC_b = {dB1_calfL,   dB1_calfL,   dB2_calfL,   dB2_calfL  };

% Set A tables
act_fi_a  = {iA_PFp_foot, iA_PFn_foot, iA_IEp_foot, iA_IEn_foot};
act_ci_a  = {iA_PFp_calf, iA_PFn_calf, iA_IEp_calf, iA_IEn_calf};
act_dtF_a = {dA2_foot,    dA2_foot,    dA1_foot,    dA1_foot   };
act_dtC_a = {dA2_calfL,   dA2_calfL,   dA1_calfL,   dA1_calfL  };
act_dpF_a = {dA1_foot,    dA1_foot,    dA2_foot,    dA2_foot   };
act_dpC_a = {dA1_calfL,   dA1_calfL,   dA2_calfL,   dA2_calfL  };

fprintf('\n=== STEP 8: ACTUATOR CANDIDATE MARKERS ===\n');

% ---- Set B report --------------------------------------------------------
fprintf('\n  ── SET B (plane contains axis + CalfY) ──\n');
fprintf('  Target plane logic: near B2 → PF/DF pair;  near B1 → IE/EV pair\n');
fprintf('  d_tgt = dist to target plane;  d_prt = dist to partner plane (sign = side)\n\n');
fprintf('  ── PF/DF pair  (nearest to B2, split by B1) ──\n');
for ki = 1:4
    if ki == 3, fprintf('\n  ── IE/EV pair  (nearest to B1, split by B2) ──\n'); end
    fi = act_fi_b{ki};  ci = act_ci_b{ki};
    fn = 'N/A';  fd_t = NaN;  fd_p = NaN;
    cn = 'N/A';  cd_t = NaN;  cd_p = NaN;
    if fi, fn = vFoot_names{fi};   fd_t = act_dtF_b{ki}(fi); fd_p = act_dpF_b{ki}(fi); end
    if ci, cn = vCalfL_names{ci};  cd_t = act_dtC_b{ki}(ci); cd_p = act_dpC_b{ki}(ci); end
    fprintf('    %s  Foot: %-12s (d_tgt=%+5.1f mm, d_prt=%+6.1f mm)   CalfL: %-12s (d_tgt=%+5.1f mm, d_prt=%+6.1f mm)\n', ...
        act_roles{ki}, fn, fd_t, fd_p, cn, cd_t, cd_p);
end

% ---- Set A report --------------------------------------------------------
fprintf('\n  ── SET A (plane normal = rotation axis) ──\n');
fprintf('  Target plane logic: near A2 → PF/DF pair;  near A1 → IE/EV pair\n');
fprintf('  d_tgt = dist to target plane;  d_prt = dist to partner plane (sign = side)\n\n');
fprintf('  ── PF/DF pair  (nearest to A2, split by A1) ──\n');
for ki = 1:4
    if ki == 3, fprintf('\n  ── IE/EV pair  (nearest to A1, split by A2) ──\n'); end
    fi = act_fi_a{ki};  ci = act_ci_a{ki};
    fn = 'N/A';  fd_t = NaN;  fd_p = NaN;
    cn = 'N/A';  cd_t = NaN;  cd_p = NaN;
    if fi, fn = vFoot_names{fi};   fd_t = act_dtF_a{ki}(fi); fd_p = act_dpF_a{ki}(fi); end
    if ci, cn = vCalfL_names{ci};  cd_t = act_dtC_a{ki}(ci); cd_p = act_dpC_a{ki}(ci); end
    fprintf('    %s  Foot: %-12s (d_tgt=%+5.1f mm, d_prt=%+6.1f mm)   CalfL: %-12s (d_tgt=%+5.1f mm, d_prt=%+6.1f mm)\n', ...
        act_roles{ki}, fn, fd_t, fd_p, cn, cd_t, cd_p);
end

fprintf('\n  Antagonist pairs for 2-actuator reduction (applies to both sets):\n');
fprintf('    PF/DF actuator: [PF+ Foot ↔ PF+ CalfL]  vs  [PF- Foot ↔ PF- CalfL]\n');
fprintf('    IE/EV actuator: [IE+ Foot ↔ IE+ CalfL]  vs  [IE- Foot ↔ IE- CalfL]\n');

%% =========================================================================
%  FIGURE 8 — Set A: planes whose NORMAL is a rotation axis  [white bg]
%             Left subplot : 3-D view (matches NSGA-II Fig1 layout)
%             Right subplot: top view in X-Z plane
% =========================================================================
figure('Name','Fig8: Set A — Planes normal to rotation axes', ...
       'Color','w', 'Position',[60 60 1300 580]);

% ---- Left: 3-D view ------------------------------------------------------
hA3 = subplot(1,2,1);
set(hA3, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k', ...
         'DataAspectRatio',[1 1 1]); hold on; grid on;
xlabel(hA3, 'X ant [mm]'); ylabel(hA3, 'Y sup [mm]'); zlabel(hA3, 'Z lat [mm]');
title(hA3, sprintf('%s — Set A  |  A1 \\perp n_{PF}  |  A2 \\perp n_{IE}', SUBJECT_NAME), ...
      'FontWeight','bold', 'FontSize',10, 'Color','k');
view(hA3, 20, 20);

% Foot markers (neutral frame, calf frame coords)
scatter3(hA3, validFoot_plan(:,1), validFoot_plan(:,2), validFoot_plan(:,3), ...
         35, [0.80 0.55 0.20], 'filled', 'DisplayName','Foot mkrs (neutral)');
% CalfL markers (neutral frame)
if ~isempty(pCalfL_cf_plan)
    scatter3(hA3, pCalfL_cf_plan(:,1), pCalfL_cf_plan(:,2), pCalfL_cf_plan(:,3), ...
             55, [0.20 0.50 0.80], 'filled', 'DisplayName','CalfL mkrs (neutral)');
end
% Joint centre
plot3(hA3, jointCentre(1), jointCentre(2), jointCentre(3), ...
      'k+', 'MarkerSize',14, 'LineWidth',2.5, 'DisplayName','Joint centre');
% Rotation axes (quivers from joint centre)
quiver3(hA3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*n_PF(1),axLen_plan*n_PF(2),axLen_plan*n_PF(3), 0, ...
        'Color',col_PF_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{PF}');
quiver3(hA3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*n_IE(1),axLen_plan*n_IE(2),axLen_plan*n_IE(3), 0, ...
        'Color',col_IE_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{IE}');
quiver3(hA3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*CalfY_cf_p(1),axLen_plan*CalfY_cf_p(2),axLen_plan*CalfY_cf_p(3), 0, ...
        'Color',col_CalfY_p, 'LineWidth',1.8, 'MaxHeadSize',0.4, 'LineStyle','--', ...
        'DisplayName','CalfY');
% Plane A1: normal = n_PF (perpendicular to PF/DF axis)
planePatch3D(hA3, jointCentre, nA1, planeHS, col_PF_p, 0.18);
planeEdge3D( hA3, jointCentre, nA1, planeHS, col_PF_p, 2.0, 'Plane A1 (\perp n_{PF})');
% Plane A2: normal = n_IE (perpendicular to Inv/Ev axis)
planePatch3D(hA3, jointCentre, nA2, planeHS, col_IE_p, 0.18);
planeEdge3D( hA3, jointCentre, nA2, planeHS, col_IE_p, 2.0, 'Plane A2 (\perp n_{IE})');
% legend(hA3, 'Location','bestoutside', 'FontSize',8);

% ---- Right: X-Z top view -------------------------------------------------
hA2 = subplot(1,2,2);
set(hA2, 'Color','w', 'XColor','k', 'YColor','k'); hold on; grid on; axis equal;
xlabel(hA2, 'X ant [mm]'); ylabel(hA2, 'Z lat [mm]');
title(hA2, 'Top view (X-Z plane)', 'FontWeight','bold', 'FontSize',10, 'Color','k');

scatter(hA2, validFoot_plan(:,1), validFoot_plan(:,3), ...
        35, [0.80 0.55 0.20], 'filled', 'DisplayName','Foot mkrs');
if ~isempty(pCalfL_cf_plan)
    scatter(hA2, pCalfL_cf_plan(:,1), pCalfL_cf_plan(:,3), ...
            55, [0.20 0.50 0.80], 'filled', 'DisplayName','CalfL mkrs');
end
plot(hA2, jointCentre(1), jointCentre(3), 'k+', ...
     'MarkerSize',14, 'LineWidth',2.5, 'DisplayName','Joint centre');
quiver(hA2, jointCentre(1),jointCentre(3), ...
       axLen_plan*n_PF(1),axLen_plan*n_PF(3), 0, ...
       'Color',col_PF_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{PF}');
quiver(hA2, jointCentre(1),jointCentre(3), ...
       axLen_plan*n_IE(1),axLen_plan*n_IE(3), 0, ...
       'Color',col_IE_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{IE}');
% Each plane appears as a line in the X-Z top view
planeLineXZ(hA2, jointCentre, nA1, planeHS, col_PF_p, 2.5, 'Plane A1');
planeLineXZ(hA2, jointCentre, nA2, planeHS, col_IE_p, 2.5, 'Plane A2');
% Actuator candidate labels (Set A proximity)
labelActuators2D(hA2, validFoot_plan, pCalfL_cf_plan, vFoot_names, vCalfL_names, ...
    {iA_PFp_foot, iA_PFn_foot, iA_IEp_foot, iA_IEn_foot}, ...
    {iA_PFp_calf, iA_PFn_calf, iA_IEp_calf, iA_IEn_calf}, jointCentre);
% legend(hA2, 'Location','bestoutside', 'FontSize',8);

sgtitle(sprintf('%s  |  SET A  |  A1: normal = n_{PF}   A2: normal = n_{IE}   |   axes angle = %.1f deg', ...
    SUBJECT_NAME, axesAngle_deg), 'FontWeight','bold', 'FontSize',11, 'Color','k');

%% =========================================================================
%  FIGURE 9 — Set B: planes CONTAINING a rotation axis + CalfY  [white bg]
%             Left subplot : 3-D view
%             Right subplot: top view in X-Z plane
% =========================================================================
figure('Name','Fig9: Set B — Planes containing axis + CalfY', ...
       'Color','w', 'Position',[80 80 1300 580]);

% ---- Left: 3-D view ------------------------------------------------------
hB3 = subplot(1,2,1);
set(hB3, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k', ...
         'DataAspectRatio',[1 1 1]); hold on; grid on;
xlabel(hB3, 'X ant [mm]'); ylabel(hB3, 'Y sup [mm]'); zlabel(hB3, 'Z lat [mm]');
title(hB3, sprintf('%s — Set B  |  B1: {n_{PF}, CalfY}  |  B2: {n_{IE}, CalfY}', SUBJECT_NAME), ...
      'FontWeight','bold', 'FontSize',10, 'Color','k');
view(hB3, 20, 20);

scatter3(hB3, validFoot_plan(:,1), validFoot_plan(:,2), validFoot_plan(:,3), ...
         35, [0.80 0.55 0.20], 'filled', 'DisplayName','Foot mkrs (neutral)');
if ~isempty(pCalfL_cf_plan)
    scatter3(hB3, pCalfL_cf_plan(:,1), pCalfL_cf_plan(:,2), pCalfL_cf_plan(:,3), ...
             55, [0.20 0.50 0.80], 'filled', 'DisplayName','CalfL mkrs (neutral)');
end
plot3(hB3, jointCentre(1), jointCentre(2), jointCentre(3), ...
      'k+', 'MarkerSize',14, 'LineWidth',2.5, 'DisplayName','Joint centre');
quiver3(hB3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*n_PF(1),axLen_plan*n_PF(2),axLen_plan*n_PF(3), 0, ...
        'Color',col_PF_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{PF}');
quiver3(hB3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*n_IE(1),axLen_plan*n_IE(2),axLen_plan*n_IE(3), 0, ...
        'Color',col_IE_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{IE}');
quiver3(hB3, jointCentre(1),jointCentre(2),jointCentre(3), ...
        axLen_plan*CalfY_cf_p(1),axLen_plan*CalfY_cf_p(2),axLen_plan*CalfY_cf_p(3), 0, ...
        'Color',col_CalfY_p, 'LineWidth',1.8, 'MaxHeadSize',0.4, 'LineStyle','--', ...
        'DisplayName','CalfY');
% Plane B1: contains n_PF and CalfY  (normal = nB1 = cross(n_PF, CalfY))
planePatch3D(hB3, jointCentre, nB1, planeHS, col_PF_p, 0.18);
planeEdge3D( hB3, jointCentre, nB1, planeHS, col_PF_p, 2.0, 'Plane B1 (n_{PF}+CalfY)');
% Plane B2: contains n_IE and CalfY  (normal = nB2 = cross(n_IE, CalfY))
planePatch3D(hB3, jointCentre, nB2, planeHS, col_IE_p, 0.18);
planeEdge3D( hB3, jointCentre, nB2, planeHS, col_IE_p, 2.0, 'Plane B2 (n_{IE}+CalfY)');
% legend(hB3, 'Location','bestoutside', 'FontSize',8);

% ---- Right: X-Z top view -------------------------------------------------
hB2 = subplot(1,2,2);
set(hB2, 'Color','w', 'XColor','k', 'YColor','k'); hold on; grid on; axis equal;
xlabel(hB2, 'X ant [mm]'); ylabel(hB2, 'Z lat [mm]');
title(hB2, 'Top view (X-Z plane)', 'FontWeight','bold', 'FontSize',10, 'Color','k');

scatter(hB2, validFoot_plan(:,1), validFoot_plan(:,3), ...
        35, [0.80 0.55 0.20], 'filled', 'DisplayName','Foot mkrs');
if ~isempty(pCalfL_cf_plan)
    scatter(hB2, pCalfL_cf_plan(:,1), pCalfL_cf_plan(:,3), ...
            55, [0.20 0.50 0.80], 'filled', 'DisplayName','CalfL mkrs');
end
plot(hB2, jointCentre(1), jointCentre(3), 'k+', ...
     'MarkerSize',14, 'LineWidth',2.5, 'DisplayName','Joint centre');
quiver(hB2, jointCentre(1),jointCentre(3), ...
       axLen_plan*n_PF(1),axLen_plan*n_PF(3), 0, ...
       'Color',col_PF_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{PF}');
quiver(hB2, jointCentre(1),jointCentre(3), ...
       axLen_plan*n_IE(1),axLen_plan*n_IE(3), 0, ...
       'Color',col_IE_p, 'LineWidth',2.5, 'MaxHeadSize',0.5, 'DisplayName','n_{IE}');
planeLineXZ(hB2, jointCentre, nB1, planeHS, col_PF_p, 2.5, 'Plane B1');
planeLineXZ(hB2, jointCentre, nB2, planeHS, col_IE_p, 2.5, 'Plane B2');
% Actuator candidate labels (Set B proximity)
labelActuators2D(hB2, validFoot_plan, pCalfL_cf_plan, vFoot_names, vCalfL_names, ...
    {iPFp_foot, iPFn_foot, iIEp_foot, iIEn_foot}, ...
    {iPFp_calf, iPFn_calf, iIEp_calf, iIEn_calf}, jointCentre);
% legend(hB2, 'Location','bestoutside', 'FontSize',8);

sgtitle(sprintf('%s  |  SET B  |  B1: {n_{PF}, CalfY}   B2: {n_{IE}, CalfY}   |   axes angle = %.1f deg', ...
    SUBJECT_NAME, axesAngle_deg), 'FontWeight','bold', 'FontSize',11, 'Color','k');

%% =========================================================================
%  STEP 9 — EXPORT ACTUATOR CANDIDATES TO OPENSIM MODELS
%
%  Two .osim files are produced, one per plane set:
%    PlaneSetA_<SUBJECT>.osim  — candidates selected by Set A proximity
%    PlaneSetB_<SUBJECT>.osim  — candidates selected by Set B proximity
%
%  Each file contains 4 PathActuator cables + 8 Markers:
%    PF+  CalfL (tibia_l)  ↔  Foot (calcn_l)
%    PF-  CalfL (tibia_l)  ↔  Foot (calcn_l)
%    IE+  CalfL (tibia_l)  ↔  Foot (calcn_l)
%    IE-  CalfL (tibia_l)  ↔  Foot (calcn_l)
%
%  Coordinate transform (MoCap calf frame mm → OpenSim body frame m):
%    tibia_l  — Kabsch via 4 anchors: Ankle3, Ankle5, Knee1, Knee2
%               Knee1/Knee2 are near the proximal tibia; Ankle3/Ankle5 are
%               near the distal tibia — excellent longitudinal spread giving
%               a well-conditioned 4-point Kabsch (no synthetic point needed)
%    calcn_l  — Kabsch via Ankle1, Ankle2, Ankle4
% =========================================================================

fprintf('\n=== STEP 9: EXPORTING CANDIDATES TO OPENSIM MODELS ===\n');

osimBase    = '/Users/rv315/Downloads/Ankle Exo Optimisation/OpenSourceResources/ankleModel_MoCapAnalysis_leftFoot_scaled.osim';
osimSetA    = sprintf('/Users/rv315/Downloads/Ankle Exo Optimisation/OpenSourceResources/ankleModel_MoCapAnalysis_leftFoot_scaled_PlaneSetA_%s.osim', SUBJECT_NAME);
osimSetB    = sprintf('/Users/rv315/Downloads/Ankle Exo Optimisation/OpenSourceResources/ankleModel_MoCapAnalysis_leftFoot_scaled_PlaneSetB_%s.osim', SUBJECT_NAME);

% ---- Anchor marker indices (all present in markerNames from CSV header) --
iAnkle1 = find(strcmp(markerNames,'Ankle1'),1);
iAnkle2 = find(strcmp(markerNames,'Ankle2'),1);
iAnkle4 = find(strcmp(markerNames,'Ankle4'),1);
assert(~isempty(iAnkle1) && ~isempty(iAnkle2) && ~isempty(iAnkle4), ...
    'Ankle1/2/4 not found in marker names — needed for calcn_l Kabsch transform.');
assert(~isempty(iKnee1) && ~isempty(iKnee2), ...
    'Knee1/Knee2 not found in marker names — needed for tibia_l Kabsch transform.');

% Neutral-frame positions in calf frame [m]
A3_cf  = p0_allMarkers_inCalf(iAnkle3,:) / 1000;
A5_cf  = p0_allMarkers_inCalf(iAnkle5,:) / 1000;
K1_cf  = p0_allMarkers_inCalf(iKnee1,:)  / 1000;
K2_cf  = p0_allMarkers_inCalf(iKnee2,:)  / 1000;
A1_cf  = p0_allMarkers_inCalf(iAnkle1,:) / 1000;
A2_cf  = p0_allMarkers_inCalf(iAnkle2,:) / 1000;
A4_cf  = p0_allMarkers_inCalf(iAnkle4,:) / 1000;

% Known positions in OpenSim tibia_l frame [m] — read from updated .osim
% Knee1 and Knee2 are now anchored to tibia_l (proximal end)
% Ankle3 and Ankle5 are at the distal tibia — full-length span of the bone
A3_tib = [ 0.024172873252011728,  -0.35906181155484157,  0.025756272370004885];
A5_tib = [-0.00078471896153142762, -0.37336169666287078, -0.044052676304729949];
K1_tib = [ 0.00098876452568420585,  0.0015467433156108323, -0.050367666573617888];
K2_tib = [-0.00067243570962066158, -0.0022476907095620446,  0.052771041718280877];

% Known positions in OpenSim calcn_l frame [m]
A1_cal = [ 0.15316854961942783,   0.018914220590540931, -0.058378868296350334];
A2_cal = [ 0.18598863569795099,   0.035827612469631093,  0.030236603249136768];
A4_cal = [ 0.013621289561055461,  0.027248322634683325,  0.014237318820520067];

% Kabsch rigid transforms — 4-point for tibia_l, 3-point for calcn_l
[R_tib,  t_tib]  = kabschRigid([A3_cf; A5_cf; K1_cf; K2_cf], ...
                                [A3_tib; A5_tib; K1_tib; K2_tib]);
[R_calc, t_calc] = kabschRigid([A1_cf; A2_cf; A4_cf], ...
                                [A1_cal; A2_cal; A4_cal]);

% Residual check
fprintf('  Kabsch residuals (should be < 3 mm):\n');
anchors_cf  = {A3_cf,  A5_cf,  K1_cf,  K2_cf,  A1_cf,   A2_cf,   A4_cf  };
anchors_tgt = {A3_tib, A5_tib, K1_tib, K2_tib, A1_cal,  A2_cal,  A4_cal };
anchors_R   = {R_tib,  R_tib,  R_tib,  R_tib,  R_calc,  R_calc,  R_calc };
anchors_t   = {t_tib,  t_tib,  t_tib,  t_tib,  t_calc,  t_calc,  t_calc };
anchors_lbl = {'Ankle3(tib)','Ankle5(tib)','Knee1(tib)','Knee2(tib)', ...
               'Ankle1(cal)','Ankle2(cal)','Ankle4(cal)'};
for ki = 1:5
    pred = (anchors_R{ki} * anchors_cf{ki}' + anchors_t{ki})';
    fprintf('    %-14s  err = %.2f mm\n', anchors_lbl{ki}, norm(pred - anchors_tgt{ki})*1000);
end

% Helper: calf frame [mm] → OpenSim body frame [m]
cfToBody = @(p_mm, R, t) (R * (p_mm(:)/1000) + t)';

% Role labels, colours, and candidate index sets for each plane set
roleNames  = {'PF_plus','PF_minus','IE_plus','IE_minus'};
roleCols   = {[0.88 0.18 0.02],[0.40 0.08 0.02],[0.05 0.35 0.82],[0.01 0.15 0.50]};

setA_fi = {iA_PFp_foot, iA_PFn_foot, iA_IEp_foot, iA_IEn_foot};
setA_ci = {iA_PFp_calf, iA_PFn_calf, iA_IEp_calf, iA_IEn_calf};
setB_fi = {iPFp_foot,   iPFn_foot,   iIEp_foot,   iIEn_foot  };
setB_ci = {iPFp_calf,   iPFn_calf,   iIEp_calf,   iIEn_calf  };

setFi   = {setA_fi, setB_fi};
setCi   = {setA_ci, setB_ci};
outFiles = {osimSetA, osimSetB};
setLabel = {'Set A (perp to axis)', 'Set B (contains axis + CalfY)'};

import org.opensim.modeling.*

for setIdx = 1:2

    fi_cell = setFi{setIdx};
    ci_cell = setCi{setIdx};

    % Convert candidate positions to OpenSim body frames [m]
    shoe_osim = zeros(4,3);   % calcn_l (foot)
    cuff_osim = zeros(4,3);   % tibia_l (CalfL)
    valid     = false(4,1);

    for ki = 1:4
        fi = fi_cell{ki};  ci = ci_cell{ki};
        if fi && ci
            shoe_osim(ki,:) = cfToBody(validFoot_plan(fi,:),  R_calc, t_calc);
            cuff_osim(ki,:) = cfToBody(pCalfL_cf_plan(ci,:),  R_tib,  t_tib);
            valid(ki)       = true;
        end
    end

    % Report
    fprintf('\n  ── %s ──\n', setLabel{setIdx});
    fprintf('  %-10s  %-12s  %-38s  %-12s  %-38s\n', ...
            'Role','Foot mkr','calcn_l [m]','CalfL mkr','tibia_l [m]');
    for ki = 1:4
        fi = fi_cell{ki};  ci = ci_cell{ki};
        fn = 'N/A';  cn = 'N/A';
        if fi, fn = vFoot_names{fi};  end
        if ci, cn = vCalfL_names{ci}; end
        if valid(ki)
            fprintf('  %-10s  %-12s  [%+.4f %+.4f %+.4f]  %-12s  [%+.4f %+.4f %+.4f]\n', ...
                roleNames{ki}, fn, shoe_osim(ki,:), cn, cuff_osim(ki,:));
        else
            fprintf('  %-10s  SKIPPED (candidate not found)\n', roleNames{ki});
        end
    end

    % Build OpenSim model
    model = Model(osimBase);
    model.setName(sprintf('AnkleExo_PlaneSet%s_%s', char('A'+setIdx-1), SUBJECT_NAME));

    bodySet = model.getBodySet();
    calcn_l = bodySet.get('calcn_l');
    tibia_l = bodySet.get('tibia_l');

    for ki = 1:4
        if ~valid(ki), continue; end
        sl  = shoe_osim(ki,:);
        cl  = cuff_osim(ki,:);
        col = roleCols{ki};
        rn  = roleNames{ki};

        % Markers
        model.addMarker(Marker(sprintf('Shoe_%s',rn), calcn_l, Vec3(sl(1),sl(2),sl(3))));
        model.addMarker(Marker(sprintf('Cuff_%s',rn), tibia_l, Vec3(cl(1),cl(2),cl(3))));

        % PathActuator cable
        cable = PathActuator();
        cable.setName(sprintf('Cable_%s',rn));
        cable.setOptimalForce(500);
        cable.setMinControl(0);
        cable.setMaxControl(1);
        gpath = cable.updGeometryPath();
        app   = Appearance();
        app.set_color(Vec3(col(1),col(2),col(3)));
        gpath.set_Appearance(app);
        gpath.appendNewPathPoint(sprintf('shoe_%s',rn), calcn_l, Vec3(sl(1),sl(2),sl(3)));
        gpath.appendNewPathPoint(sprintf('cuff_%s',rn), tibia_l, Vec3(cl(1),cl(2),cl(3)));
        model.addForce(cable);
    end

    model.finalizeConnections();
    model.print(outFiles{setIdx});
    nValid = sum(valid);
    fprintf('  -> %d markers + %d PathActuator cables saved to:\n     %s\n', ...
            2*nValid, nValid, outFiles{setIdx});
end

fprintf('\nDone. Open the two .osim files in OpenSim GUI to inspect cable routing.\n');

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

% ---- Draw filled semi-transparent plane patch (3-D) ----------------------
%  q        : 1x3 point on the plane
%  n        : 1x3 plane normal (normalised internally)
%  halfSize : half-edge length of the square patch [mm]
%  col      : 1x3 RGB colour
%  alpha    : face transparency (0 = invisible, 1 = opaque)
function planePatch3D(hAx, q, n, halfSize, col, alpha)
    n = n(:)' / norm(n);
    if abs(n(1)) < 0.9
        u_raw = cross(n, [1 0 0]);
    else
        u_raw = cross(n, [0 1 0]);
    end
    u = u_raw / norm(u_raw);
    v = cross(n, u);  v = v / norm(v);
    c = [q + halfSize*u + halfSize*v;
         q - halfSize*u + halfSize*v;
         q - halfSize*u - halfSize*v;
         q + halfSize*u - halfSize*v];
    patch(hAx, 'XData', c(:,1), 'YData', c(:,2), 'ZData', c(:,3), ...
          'FaceColor', col, 'FaceAlpha', alpha, 'EdgeColor', 'none');
end

% ---- Draw boundary edge of a plane patch (adds a legend entry) -----------
function planeEdge3D(hAx, q, n, halfSize, col, lw, label)
    n = n(:)' / norm(n);
    if abs(n(1)) < 0.9
        u_raw = cross(n, [1 0 0]);
    else
        u_raw = cross(n, [0 1 0]);
    end
    u = u_raw / norm(u_raw);
    v = cross(n, u);  v = v / norm(v);
    c = [q + halfSize*u + halfSize*v;
         q - halfSize*u + halfSize*v;
         q - halfSize*u - halfSize*v;
         q + halfSize*u - halfSize*v;
         q + halfSize*u + halfSize*v];   % close the loop
    plot3(hAx, c(:,1), c(:,2), c(:,3), '-', ...
          'Color', col, 'LineWidth', lw, 'DisplayName', label);
end

% ---- Intersection of a plane with the X-Z view (top view) ----------------
%  A plane through q with normal n intersects the horizontal slice y = q(2)
%  as the line:  n(1)*(x-q(1)) + n(3)*(z-q(3)) = 0
%  Direction in (X, Z):  d = [-n(3),  n(1)]
function planeLineXZ(hAx, q, n, halfSize, col, lw, label)
    n  = n(:)';
    d  = [-n(3), n(1)];      % direction in (X, Z) coordinate space
    dn = norm(d);
    if dn < 1e-8
        % Normal is purely along Y: plane is horizontal, appears as a dot
        plot(hAx, q(1), q(3), 'x', 'Color',col, 'MarkerSize',10, ...
             'LineWidth',lw, 'DisplayName',label);
        return;
    end
    d  = d / dn;
    x1 = q(1) - halfSize*d(1);  z1 = q(3) - halfSize*d(2);
    x2 = q(1) + halfSize*d(1);  z2 = q(3) + halfSize*d(2);
    plot(hAx, [x1 x2], [z1 z2], '-', 'Color',col, 'LineWidth',lw, 'DisplayName',label);
end

% ---- Find closest marker to a target plane on one side of partner plane --
%  dist_target  : Nx1 signed distances from target plane (minimise |val|)
%  dist_partner : Nx1 signed distances from partner plane (side selector)
%  side         : +1 → positive side of partner,  -1 → negative side
%  Returns: dist = absolute dist to target plane,  idx = row index (0 if none)
function [dist, idx] = findClosestOnSide(dist_target, dist_partner, side)
    % side: +1 → positive side,  -1 → negative side
    % Use element-wise multiplication to avoid && on vectors
    mask = (dist_partner * side) > 0;
    if ~any(mask)
        % Fall back: include markers sitting exactly on the partner plane
        mask = (dist_partner * side) >= 0;
    end
    if ~any(mask)
        dist = Inf;  idx = 0;  return;
    end
    [dist, rel_idx] = min(abs(dist_target(mask)));
    all_idx = find(mask);
    idx     = all_idx(rel_idx);
end

% ---- Draw actuator candidate labels on a 2-D XZ top-view axis -----------
%  foot_pos    : Kf×3 valid foot marker positions (calf frame)
%  calf_pos    : Kc×3 valid CalfL marker positions (calf frame)
%  foot_names  : 1×Kf cell of foot marker names
%  calf_names  : 1×Kc cell of CalfL marker names
%  fi_cell     : {PF+ PF- IE+ IE-} foot indices (0 = not found)
%  ci_cell     : {PF+ PF- IE+ IE-} CalfL indices (0 = not found)
%  jc          : 1×3 joint centre (calf frame) — used for outward text offset
function labelActuators2D(hAx, foot_pos, calf_pos, foot_names, calf_names, fi_cell, ci_cell, jc)
    role_lbl  = {'PF+', 'PF-', 'IE+', 'IE-'};
    role_col  = {[0.88 0.18 0.02],  ...  % PF+  warm red
                 [0.40 0.08 0.02],  ...  % PF-  dark red
                 [0.05 0.35 0.82],  ...  % IE+  bright blue
                 [0.01 0.15 0.50]};      % IE-  dark blue

    jcXZ = [jc(1), jc(3)];   % joint centre in XZ
    txtDist = 14;             % mm — label stand-off from marker centre

    for ki = 1:4
        col = role_col{ki};
        lbl = role_lbl{ki};
        fi  = fi_cell{ki};
        ci  = ci_cell{ki};

        % ---- Cable line: Foot ↔ CalfL for this role ----------------------
        if fi && ci
            fx = foot_pos(fi,1);  fz = foot_pos(fi,3);
            cx = calf_pos(ci,1);  cz = calf_pos(ci,3);
            plot(hAx, [fx cx], [fz cz], '--', 'Color', [col 0.55], ...
                 'LineWidth', 1.4, 'HandleVisibility','off');
        end

        % ---- Foot marker (circle) ----------------------------------------
        if fi
            px = foot_pos(fi,1);  pz = foot_pos(fi,3);
            scatter(hAx, px, pz, 220, col, 'o', ...
                    'LineWidth', 2.8, 'HandleVisibility','off');
            % Offset direction: away from joint centre in XZ
            dv  = [px - jcXZ(1), pz - jcXZ(2)];
            dv  = dv / max(norm(dv), 1e-6);
            tx  = px + txtDist * dv(1);
            tz  = pz + txtDist * dv(2);
            ha  = ternary(dv(1) >= 0, 'left', 'right');
            text(hAx, tx, tz, {lbl, foot_names{fi}}, ...
                 'Color', col, 'FontSize', 8, 'FontWeight', 'bold', ...
                 'Interpreter','none', 'HorizontalAlignment', ha);
        end

        % ---- CalfL marker (square) ----------------------------------------
        if ci
            px = calf_pos(ci,1);  pz = calf_pos(ci,3);
            scatter(hAx, px, pz, 220, col, 's', ...
                    'LineWidth', 2.8, 'HandleVisibility','off');
            dv  = [px - jcXZ(1), pz - jcXZ(2)];
            dv  = dv / max(norm(dv), 1e-6);
            tx  = px + txtDist * dv(1);
            tz  = pz + txtDist * dv(2);
            ha  = ternary(dv(1) >= 0, 'left', 'right');
            text(hAx, tx, tz, {lbl, calf_names{ci}}, ...
                 'Color', col, 'FontSize', 8, 'FontWeight', 'bold', ...
                 'Interpreter','none', 'HorizontalAlignment', ha);
        end
    end

    % Legend patch for actuator roles
    for ki = 1:4
        scatter(hAx, nan, nan, 100, role_col{ki}, 'o', 'LineWidth', 2.2, ...
                'DisplayName', sprintf('%s (Foot=○  CalfL=□)', role_lbl{ki}));
    end
    legend(hAx, 'show', 'Location','bestoutside', 'FontSize', 7);
end

% ---- Ternary helper (avoids inline if in function arguments) -------------
function v = ternary(cond, a, b)
    if cond, v = a; else, v = b; end
end

% ---- Kabsch rigid-body transform -----------------------------------------
%  Finds R (3×3) and t (3×1) such that:  tgt ≈ (R * src' + t)'
%  src, tgt : [N×3] corresponding point sets (at least 3 non-collinear pts)
function [R, t] = kabschRigid(src, tgt)
    mu_s = mean(src,1);  mu_t = mean(tgt,1);
    A    = (src - mu_s)' * (tgt - mu_t);
    [U,~,V] = svd(A);
    d = sign(det(V*U'));
    R = V * diag([1,1,d]) * U';   % maps src column-vec → tgt column-vec
    t = mu_t' - R*mu_s';           % [3×1]
end

% ---- CSV path helper -----------------------------------------------------
function p = csvPathFor(baseDir, subject, trial)
    d = fullfile(baseDir, subject, trial);
    f = dir(fullfile(d, '*.csv'));
    if isempty(f), error('No CSV found in: %s', d); end
    p = fullfile(d, f(1).name);
end

% ---- Load full trial (header + data) -------------------------------------
function [markerNames, markerXCols, markerData, frameNumbers] = loadTrialCSV(csvPath)
    fprintf('Loading: %s\n', csvPath);
    hdrCells    = readcell(csvPath, 'Range','3:3', 'Delimiter',',');
    markerNames = {};  markerXCols = [];
    for c = 1:numel(hdrCells)
        nm = string(hdrCells{c});
        if ismissing(nm) || strtrim(nm) == "", continue; end
        tok = strsplit(char(nm), ':');
        markerNames{end+1} = strtrim(tok{end}); %#ok<AGROW>
        markerXCols(end+1) = c;                 %#ok<AGROW>
    end
    raw          = readmatrix(csvPath, 'NumHeaderLines',5, 'OutputType','double');
    markerData   = unpackMarkers(raw, markerXCols);
    frameNumbers = raw(:,1);
end

% ---- Load data only (reuse known column indices) -------------------------
function markerData = loadTrialData(csvPath, markerXCols)
    fprintf('Loading: %s\n', csvPath);
    raw        = readmatrix(csvPath, 'NumHeaderLines',5, 'OutputType','double');
    markerData = unpackMarkers(raw, markerXCols);
end

% ---- Unpack raw matrix -> (nFrames x nMarkers x 3) ----------------------
function md = unpackMarkers(raw, markerXCols)
    nF = size(raw,1);  nM = numel(markerXCols);
    md = nan(nF, nM, 3);
    for m = 1:nM
        cx        = markerXCols(m);
        md(:,m,1) = raw(:,cx);
        md(:,m,2) = raw(:,cx+1);
        md(:,m,3) = raw(:,cx+2);
    end
end

% ---- Safe normalise ------------------------------------------------------
function v = snorm(v)
    n = norm(v);
    if n > 1e-10, v = v/n; else, v(:) = 0; end
end

% ---- Rotation matrix -> rotation vector (matrix log map) ----------------
function rv = rotmat2vec(R)
    cosTheta = max(-1, min(1, (trace(R)-1)/2));
    theta    = acos(cosTheta);
    if theta < 1e-6
        rv = zeros(1,3); return;
    end
    if abs(theta - pi) < 1e-4
        rv = []; return;
    end
    axis = [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)] / (2*sin(theta));
    rv   = (theta * axis)';
end

% ---- Rodrigues rotation matrix -------------------------------------------
%  R = rodrigues(n, theta)
%  n     : 3-element unit axis vector (row or column)
%  theta : angle in radians
function R = rodrigues(n, theta)
    n = n(:) / norm(n);
    K = [  0    -n(3)  n(2);
          n(3)   0    -n(1);
         -n(2)  n(1)   0  ];
    R = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
end

% ---- Exact sequential angle decomposition (Grood-Suntay / JCS) ----------
%
%  Solves exactly:  R(n_PF, theta_PF) * R(n_IE_body, theta_IE) = R_rel
%
%  Convention:
%    PF/DF is proximal (calf-fixed) axis — applied first.
%    IE/EV is distal   (foot-fixed)  axis — applied second.
%    At neutral (R_rel = I): theta_PF = theta_IE = 0 exactly.
%
%  Method (floating-axis):
%    e1    = n_PF                          (fixed in calf frame)
%    e3    = R_rel * n_IE                  (IE axis, currently in calf frame)
%    e2    = cross(e1, e3) / norm(...)     (floating axis, perp to both)
%    e2_0  = cross(n_PF, n_IE) / norm(...) (floating axis at neutral)
%
%    theta_PF: angle from e2_0 to e2 about n_PF
%    theta_IE: angle from e2_0 to e2_foot about n_IE,
%              where e2_foot = R_rel' * e2  (floating axis in foot frame)
%
%  Singularity: n_PF || e3, i.e. theta_IE = +-90 deg — anatomically
%  unreachable for the ankle; returns NaN in that case.
%
function [theta_PF, theta_IE] = decomposeAnglesExact(R_rel, n_PF, n_IE)
    n_PF = n_PF(:);
    n_IE = n_IE(:);

    % Current IE axis expressed in calf frame
    e3 = R_rel * n_IE;

    % Floating axis at neutral (reference): cross(n_PF, n_IE)
    e2_0_raw  = cross(n_PF, n_IE);
    e2_0_norm = norm(e2_0_raw);
    if e2_0_norm < 1e-6
        % n_PF and n_IE are parallel — degenerate axes definition
        theta_PF = nan;  theta_IE = nan;  return;
    end
    e2_0 = e2_0_raw / e2_0_norm;

    % Current floating axis: cross(n_PF, e3)
    e2_raw  = cross(n_PF, e3);
    e2_norm = norm(e2_raw);
    if e2_norm < 1e-6
        % n_PF || e3 → theta_IE = +-90 deg — singular pose
        theta_PF = nan;  theta_IE = nan;  return;
    end
    e2 = e2_raw / e2_norm;

    % theta_PF: signed angle from e2_0 to e2, measured about n_PF
    %   sin(theta_PF) component along n_PF = dot(n_PF, cross(e2_0, e2))
    %   cos(theta_PF)                      = dot(e2_0, e2)
    theta_PF = atan2( dot(n_PF, cross(e2_0, e2)),  dot(e2_0, e2) ) * 180/pi;

    % theta_IE: signed angle from e2_0 to e2_foot, measured about n_IE
    %   e2_foot = R_rel' * e2  (floating axis expressed in foot frame)
    %   At neutral R_rel=I so e2_foot = e2 = e2_0 and theta_IE = 0.
    e2_foot = R_rel' * e2;
    theta_IE = atan2( dot(n_IE, cross(e2_0, e2_foot)),  dot(e2_0, e2_foot) ) * 180/pi;
end

% ---- Compute calf frame per frame ----------------------------------------
function [O_calf, R_calf, calfVec] = computeCalfFrames(md, iA3, iA5, iK1, iK2, iU, iL)
    nF = size(md,1);
    O_calf  = zeros(nF,3);
    R_calf  = zeros(3,3,nF);
    calfVec = zeros(nF,3);
    for f = 1:nF
        pA3 = squeeze(md(f,iA3,:))';  pA5 = squeeze(md(f,iA5,:))';
        pK1 = squeeze(md(f,iK1,:))';  pK2 = squeeze(md(f,iK2,:))';
        pU  = squeeze(md(f,iU,:));    pL  = squeeze(md(f,iL,:));
        mU  = mean(pU(~any(isnan(pU),2),:), 1);
        mL  = mean(pL(~any(isnan(pL),2),:), 1);
        if any(isnan([pA3 pA5 pK1 pK2 mU mL]))
            if f > 1
                O_calf(f,:)   = O_calf(f-1,:);
                R_calf(:,:,f) = R_calf(:,:,f-1);
                calfVec(f,:)  = calfVec(f-1,:);
            end
            continue;
        end
        O_calf(f,:)   = 0.5*(pA3 + pA5);
        Zc            = snorm(pK2 - pK1);
        cv            = snorm(mU  - mL);
        Xc            = snorm(cross(cv, Zc));
        Yc            = snorm(cross(Zc, Xc));
        R_calf(:,:,f) = [Xc', Yc', Zc'];
        calfVec(f,:)  = cv;
    end
end

% ---- Compute foot frame per frame (SVD plane fit) ------------------------
function [O_foot, R_foot, footNormal] = computeFootFrames(md, iFoot, calfVec)
    nF         = size(md,1);
    O_foot     = zeros(nF,3);
    R_foot     = zeros(3,3,nF);
    footNormal = zeros(nF,3);
    prevLong   = [];
    for f = 1:nF
        pts   = squeeze(md(f,iFoot,:));
        valid = ~any(isnan(pts),2);
        pts   = pts(valid,:);
        if size(pts,1) < 3
            if f > 1
                O_foot(f,:)     = O_foot(f-1,:);
                R_foot(:,:,f)   = R_foot(:,:,f-1);
                footNormal(f,:) = footNormal(f-1,:);
            end
            continue;
        end
        cen  = mean(pts,1);
        O_foot(f,:) = cen;
        [~,~,V] = svd(pts - cen, 'econ');
        fNorm = V(:,3)';  fLong = V(:,1)';
        if dot(fNorm, calfVec(f,:)) < 0, fNorm = -fNorm; end
        if ~isempty(prevLong) && dot(fLong, prevLong) < 0, fLong = -fLong; end
        prevLong = fLong;
        fCross   = snorm(cross(fNorm, fLong));
        fLong    = snorm(cross(fCross, fNorm));
        R_foot(:,:,f)   = [fLong', fCross', fNorm'];
        footNormal(f,:) = fNorm;
    end
end

% ---- Draw RGB coordinate triad -------------------------------------------
function drawTriad(hAx, origin, R, scale, label)
    triCols = {[1 0.2 0.2],[0.2 1 0.2],[0.4 0.6 1]};
    names   = {'X','Y','Z'};
    for i = 1:3
        d   = R(:,i)';
        tip = origin + scale*d;
        plot3(hAx,[origin(1) tip(1)],[origin(2) tip(2)],[origin(3) tip(3)],'Color',triCols{i},'LineWidth',2.5);
        text(hAx,tip(1),tip(2),tip(3),sprintf('%s_%s',names{i},label),'Color',triCols{i},'FontSize',7,'FontWeight','bold','Interpreter','none');
    end
end

% ---- Draw one rotation axis -----------------------------------------------
function drawRotAxis(hAx, q, n, halfLen, col, label)
    p1 = q - halfLen*n;  p2 = q + halfLen*n;
    plot3(hAx,[p1(1) p2(1)],[p1(2) p2(2)],[p1(3) p2(3)],'-','Color',col,'LineWidth',4);
    text(hAx,p2(1),p2(2),p2(3)+4,label,'Color',col,'FontSize',9,'FontWeight','bold','Interpreter','none');
end

% ---- Render one frame for Figure 5 animation -----------------------------
function renderAxisFrame(hAx, dispData, markerNames, markerColours, ...
                         fIdx, nFTotal, subjectName, trialName, ...
                         n_PF, q_PF, n_IE, q_IE, jointCentre, ...
                         footTriadR, ang_PF_V, ang_IE_V, ...
                         axisScale, showLabels, ...
                         iPF_pos, iPF_neg, iIE_pos, iIE_neg, iFoot, p0_foot_all)
    cla(hAx);
    xl = hAx.XLim;  yl = hAx.YLim;  zl = hAx.ZLim;
    hAngText = text(hAx, xl(1)+0.02*(xl(2)-xl(1)), yl(2)-0.02*(yl(2)-yl(1)), zl(2)-0.02*(zl(2)-zl(1)), '', ...
                'Color','w','FontSize',12,'FontWeight','bold','VerticalAlignment','top','Interpreter','none');
    t_s = (fIdx-1) / 100;
    title(hAx, sprintf('%s  —  %s  [calf frame]  |  Frame %d / %d  (t = %.2f s)', ...
          subjectName, trialName, fIdx, nFTotal, t_s), 'Color','w','FontSize',10,'Interpreter','none');
    nM = size(dispData,2);
    bestGlobalIdx = iFoot([iPF_pos, iPF_neg, iIE_pos, iIE_neg]);
    bestCols  = {[1.0 0.92 0.2],[1.0 0.55 0.1],[0.15 1.0 0.85],[0.05 0.65 1.0]};
    bestLabels = {'+PF','-PF','+IE','-IE'};
    for m_ = 1:nM
        x = dispData(fIdx,m_,1);  y = dispData(fIdx,m_,2);  z = dispData(fIdx,m_,3);
        if isnan(x)||isnan(y)||isnan(z), continue; end
        bIdx = find(bestGlobalIdx == m_, 1);
        if ~isempty(bIdx)
            col_ = bestCols{bIdx};
            scatter3(hAx,x,y,z,260,col_,'o','LineWidth',2.2,'MarkerEdgeColor',col_,'MarkerFaceColor','none');
            scatter3(hAx,x,y,z,110,col_,'filled','MarkerEdgeColor','w','LineWidth',0.8);
            if showLabels
                text(hAx,x,y,z+14,sprintf('%s\\n%s',markerNames{m_},bestLabels{bIdx}),'Color',col_,'FontSize',7,'HorizontalAlignment','center','FontWeight','bold','Interpreter','none');
            end
        else
            col_ = markerColours(m_,:);
            scatter3(hAx,x,y,z,90,col_,'filled','MarkerEdgeColor','w','LineWidth',0.4);
            if showLabels
                text(hAx,x,y,z+12,markerNames{m_},'Color','w','FontSize',6,'HorizontalAlignment','center','BackgroundColor',[0.08 0.08 0.08],'EdgeColor',col_,'Margin',1,'Interpreter','none');
            end
        end
    end
    allBestIdx = [iPF_pos, iPF_neg, iIE_pos, iIE_neg];
    for bi = 1:4
        p = p0_foot_all(allBestIdx(bi),:);
        if any(isnan(p)), continue; end
        col_ = bestCols{bi};
        plot3(hAx,[jointCentre(1) p(1)],[jointCentre(2) p(2)],[jointCentre(3) p(3)],'--','Color',[col_ 0.55],'LineWidth',1.2);
    end
    drawTriad(hAx,[0 0 0],eye(3),axisScale,'Calf');
    drawTriad(hAx,[0 0 0],footTriadR(:,:,fIdx),axisScale*0.8,'Foot');
    drawRotAxis(hAx,q_PF,n_PF,80,[1 0.5 0.1],'PF/DF');
    drawRotAxis(hAx,q_IE,n_IE,80,[0.2 0.8 1],'Inv/Ev');
    scatter3(hAx,jointCentre(1),jointCentre(2),jointCentre(3),220,[1 0.9 0],'p','filled','MarkerEdgeColor','w','LineWidth',1.2);
    set(hAngText,'String',sprintf('PF/DF (exact):  %+.1f deg\nInv/Ev (exact): %+.1f deg',ang_PF_V(fIdx),ang_IE_V(fIdx)));
    drawnow;
end
%% =========================================================================
