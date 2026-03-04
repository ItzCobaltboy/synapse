clear all; clc;

port = 'COM5';
baud = 115200;
s = serialport(port, baud);

WINDOW = 100;
DURATION = 5;   % seconds per gesture
GESTURES = {'rest', 'g1', 'g2', 'g3', 'g4'};

raw = struct();

disp("Starting calibration recording...");

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
    fprintf("Collected %d samples\n", size(samples,1));
end

save calibration_raw.mat raw
disp("Saved calibration_raw.mat");
