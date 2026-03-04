load calibration_raw.mat

gestures = fieldnames(raw);

% Personal channel mean/std from REST
rest = raw.rest;

personal_mean = mean(rest,1);
personal_std = std(rest,[],1);

% Gain normalization per channel
gain = 1 ./ max(abs(rest),[],1);

% Build class-remap using the MATLAB feature extractor + Python server
url = 'http://127.0.0.1:8000/predict';

class_map = containers.Map();

% For gestures g1..g4 (indices 1..4)
for true_label = 1:4
    name = ['g' num2str(true_label)];
    samples = raw.(name);

    preds = [];

    % slide window of WINDOW=100
    WINDOW = 100;

    for i = 1:size(samples,1)-WINDOW
        buf1 = samples(i:i+WINDOW-1, 1);
        buf2 = samples(i:i+WINDOW-1, 2);

        feats1 = int_feature(buf1);
        feats2 = int_feature(buf2);
        feats = [feats1 feats2];
        feats = double(feats(:)');

        % JSON
        payload = struct("features", feats);
        rawJSON = jsonencode(payload);
        rawJSON = strrep(rawJSON, 'null', '0');

        opts = weboptions("MediaType","application/json", "Timeout",2, "RequestMethod","post");

        try
            r = webwrite(url, rawJSON, opts);
            preds(end+1) = r.gesture;
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
fid = fopen('user_calibration.json','w');
fprintf(fid, "%s", json);
fclose(fid);

disp("Calibration complete, saved user_calibration.json.");
