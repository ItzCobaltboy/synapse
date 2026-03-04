clear all; clc;

% ==============================
% Resolve script-relative paths
% ==============================
script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(script_dir);
matlab_rt_dir = fullfile(repo_root, "matlab_realtime");

% Ensure feature extraction code is on path
addpath(matlab_rt_dir);
addpath(fullfile(matlab_rt_dir, "EMG-Feature-Extraction-Toolbox"));

% ==============================
% Serial configuration
% ==============================
port = 'COM5';
baud = 115200;
s = serialport(port, baud);

WINDOW = 100;
DURATION = 5;   % seconds per gesture
GESTURES = {'rest', 'g1', 'g2', 'g3', 'g4'};

raw = struct();

disp("Starting calibration recording...");

% ==============================
% Step 1: Collect raw calibration data
% ==============================
for g = 1:length(GESTURES)
    name = GESTURES{g};

    fprintf("\nHold gesture '%s' for %d seconds...\n", name, DURATION);
    pause(2);

    samples = [];
    t0 = tic;

    while toc(t0) < DURATION
        line = readline(s);
        tokens = regexp(line, "<([\d\.\-]+),([\d\.\-]+)>", "tokens");
        if isempty(tokens)
            continue;
        end

        vals = str2double(tokens{1});
        samples(end+1, :) = vals; %#ok<AGROW>
    end

    raw.(name) = samples;
    fprintf("Collected %d samples\n", size(samples, 1));
end

save(fullfile(script_dir, "calibration_raw.mat"), "raw");
disp("Saved calibration_raw.mat");

% ==============================
% Step 2: Build calibration profile
% ==============================
if ~isfield(raw, 'rest') || isempty(raw.rest)
    error("Missing 'rest' samples. Cannot build calibration profile.");
end

% Personal channel mean/std from REST
rest = raw.rest;
personal_mean = mean(rest, 1);
personal_std = std(rest, [], 1);

% Avoid divide-by-zero
personal_std(personal_std == 0) = 1;

% Gain normalization per channel
gain = 1 ./ max(abs(rest), [], 1);
gain(~isfinite(gain)) = 1;

% Build class-remap using MATLAB feature extractor + Python server
url = 'http://127.0.0.1:8000/predict';
opts = weboptions("MediaType", "application/json", "Timeout", 2, "RequestMethod", "post");

class_map = containers.Map();

for true_label = 1:4
    name = ['g' num2str(true_label)];

    if ~isfield(raw, name) || isempty(raw.(name))
        warning("Gesture %s missing. Skipping.", name);
        continue;
    end

    samples = raw.(name);
    preds = [];

    for i = 1:size(samples, 1) - WINDOW
        buf1 = samples(i:i+WINDOW-1, 1);
        buf2 = samples(i:i+WINDOW-1, 2);

        feats1 = int_feature(buf1);
        feats2 = int_feature(buf2);
        feats = [feats1 feats2];
        feats = double(feats(:)');

        payload = struct("features", feats);
        rawJSON = jsonencode(payload);
        rawJSON = strrep(rawJSON, 'null', '0');

        try
            r = webwrite(url, rawJSON, opts);
            preds(end+1) = r.gesture; %#ok<AGROW>
        catch
            continue;
        end
    end

    if isempty(preds)
        mapped = -1;
    else
        mapped = mode(preds);
    end

    class_map(num2str(mapped)) = true_label;
end

% Save calibration to JSON
calib.mean = personal_mean;
calib.std = personal_std;
calib.gain = gain;
calib.class_map = class_map;

json = jsonencode(calib);
user_calib_file = fullfile(script_dir, "user_calibration.json");
fid = fopen(user_calib_file, 'w');
fprintf(fid, "%s", json);
fclose(fid);

disp("Calibration complete, saved user_calibration.json.");
