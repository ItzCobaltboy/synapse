clear all; clc;

% ==============================
% Resolve script-relative paths
% ==============================
script_dir = fileparts(mfilename("fullpath"));
repo_root = fileparts(script_dir);
calib_path = fullfile(repo_root, "calibration", "user_calibration.json");

% Ensure local functions/toolbox are reachable
addpath(script_dir);
addpath(fullfile(script_dir, "EMG-Feature-Extraction-Toolbox"));

% ==============================
% Load Personal Calibration
% ==============================
if isfile(calib_path)
    disp("Loading calibration...");
    calib = jsondecode(fileread(calib_path));
    disp("Calibration loaded.");
else
    disp("WARNING: No calibration file found. Using identity calibration.");

    % Identity calibration so the script doesn't die
    calib.mean = [0, 0];
    calib.std = [1, 1];
    calib.gain = [1, 1];
    calib.class_map = struct();
end

% ==============================
% Serial Config
% ==============================
port = 'COM5';           % CHANGE THIS
baud = 115200;
s = serialport(port, baud);
WINDOW = 100;

buffer1 = zeros(WINDOW,1);
buffer2 = zeros(WINDOW,1);
idx = 1;

disp("MATLAB realtime feature extractor + predictor running...");

% ==============================
% FastAPI Endpoint
% ==============================
url = 'http://127.0.0.1:8000/predict';
opts = weboptions( ...
    "MediaType","application/json", ...
    "Timeout", 3, ...
    "RequestMethod","post" ...
);

while true
    % Read EMG from ESP32
    line = readline(s);

    tokens = regexp(line, "<([\d\.\-]+),([\d\.\-]+)>", "tokens");
    if isempty(tokens)
        continue;
    end

    vals = str2double(tokens{1});
    v1 = vals(1);
    v2 = vals(2);

    % ==============================
    % Apply PERSONAL calibration
    % ==============================
    v1 = (v1 * calib.gain(1) - calib.mean(1)) / calib.std(1);
    v2 = (v2 * calib.gain(2) - calib.mean(2)) / calib.std(2);

    % ==============================
    % Fill buffer
    % ==============================
    buffer1(idx) = v1;
    buffer2(idx) = v2;
    idx = idx + 1;

    % ==============================
    % WINDOW READY -> EXTRACT FEATURES
    % ==============================
    if idx > WINDOW
        feats1 = int_feature(buffer1);
        feats2 = int_feature(buffer2);

        feats = [feats1 feats2];
        feats = double(feats(:)');

        % ==============================
        % Prepare JSON
        % ==============================
        payload = struct("features", feats);
        rawJSON = jsonencode(payload);
        rawJSON = strrep(rawJSON, 'null', '0');  % safety

        % DEBUG
        fprintf("Sending features (%d dims)\n", length(feats));

        % ==============================
        % Request to FastAPI
        % ==============================
        try
            response = webwrite(url, rawJSON, opts);
            fprintf("Predicted Gesture: %d\n", response.gesture);
        catch ME
            warning("POST failed: %s", ME.message);
        end

        idx = 1;  % reset window
    end
end
