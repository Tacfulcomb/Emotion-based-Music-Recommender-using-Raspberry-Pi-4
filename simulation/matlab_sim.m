%% Emotion-based system simulation in MATLAB
% Discrete-time simulation of:
% Camera -> OpenCV Preprocess -> CNN Inference -> MusicDB/Audio Output

clear; clc; close all;

%% 0. Camera & CNN configuration (for documentation / reporting)
cam_width   = 640;           % Camera resolution width (pixels) – can be 2592 for 5MP
cam_height  = 480;           % Camera resolution height (pixels) – can be 1944 for 5MP
cam_fps_in  = 25;            % Target camera input FPS (e.g. 25 FPS)

% OpenCV / CNN configuration (conceptual)
cnn_input_size  = [48 48 1];         % e.g. 48x48 grayscale face crop
cnn_backend     = 'OpenCV-DNN';     % or 'TFLite on Raspberry Pi 4';
cnn_device      = 'CPU';            % CPU-only inference
% Note: concrete latencies are defined per scenario below.


%% 1. Global simulation settings
sim_duration   = 3;                     % Simulation time (seconds)
frame_period   = 1 / cam_fps_in;        % Time between camera frames (s)
num_frames     = floor(sim_duration * cam_fps_in);

% Base delays for non-CNN stages (in milliseconds)
pre_ms_base = 10;   % OpenCV preprocess (grayscale + Haar + resize), ~10 ms/frame
db_ms_base  = 2;    % DB lookup + audio trigger, ~2 ms/frame


%% 2. Helper: simulate a single hardware scenario
simulate_scenario = @(pre_ms, inf_ms, db_ms) simulate_pipeline( ...
    num_frames, ...      % number of frames to simulate
    frame_period, ...    % time between camera frames
    pre_ms, ...          % preprocessing delay (ms)
    inf_ms, ...          % CNN latency (ms)
    db_ms);              % DB + audio delay (ms)


%% 3. Define and run three hardware scenarios
% Scenario A: Low-end board (very slow CNN)
% Example: CNN inference ~ 250 ms/frame -> ~3–4 FPS effective
resultA = simulate_scenario(pre_ms_base, 250, db_ms_base);

% Scenario B: Mid-range board
% Example: CNN inference ~ 160 ms/frame -> ~5–6 FPS effective
resultB = simulate_scenario(pre_ms_base, 160, db_ms_base);

% Scenario C: Raspberry Pi 4 (target)
% Example: CNN inference ~ 120 ms/frame -> ~7–8 FPS effective
resultC = simulate_scenario(pre_ms_base, 120, db_ms_base);


%% 4. Print results to console
fprintf('=== Scenario A: Low-end board ===\n');
print_result(resultA);

fprintf('\n=== Scenario B: Mid-range board ===\n');
print_result(resultB);

fprintf('\n=== Scenario C: Raspberry Pi 4 (target) ===\n');
print_result(resultC);


%% 5. Plot effective FPS comparison
figure;
bar([resultA.effective_fps, resultB.effective_fps, resultC.effective_fps]);
set(gca, 'XTickLabel', {'Low-end','Mid-range','Pi 4'});
ylabel('Effective FPS');
title('Comparison of effective FPS across hardware scenarios');
grid on;


%% ====== FUNCTION: simulate the end-to-end pipeline ======
function result = simulate_pipeline(num_frames, frame_period, pre_ms, inf_ms, db_ms)
    % Convert delays from milliseconds to seconds
    pre_s = pre_ms / 1000;
    inf_s = inf_ms / 1000;
    db_s  = db_ms  / 1000;

    arrival_times = zeros(num_frames,1);  % time when each frame arrives at camera
    finish_times  = zeros(num_frames,1);  % time when each frame leaves the pipeline
    latencies     = zeros(num_frames,1);  % end-to-end latency per frame

    last_finish_time = 0;

    for i = 1:num_frames
        % Time when frame i is captured by the camera
        arrival_times(i) = (i-1) * frame_period;

        % If the pipeline is busy, the frame must wait until previous frame finishes
        start_time = max(arrival_times(i), last_finish_time);

        % Total processing time for one frame through all modules
        processing_time = pre_s + inf_s + db_s;

        % Time when this frame leaves the pipeline
        finish_times(i) = start_time + processing_time;

        % Update for the next frame
        last_finish_time = finish_times(i);

        % End-to-end latency
        latencies(i) = finish_times(i) - arrival_times(i);
    end

    % Total time from first to last processed frame
    total_time    = finish_times(end);

    % Effective FPS = number of frames / total elapsed time
    effective_fps = num_frames / total_time;

    % Pack results into a struct
    result.effective_fps = effective_fps;
    result.avg_latency   = mean(latencies);
    result.max_latency   = max(latencies);
    result.total_time    = total_time;
end


%% ====== FUNCTION: nicely print the results ======
function print_result(result)
    fprintf('Effective FPS     : %.2f\n', result.effective_fps);
    fprintf('Average latency   : %.2f ms\n', result.avg_latency * 1000);
    fprintf('Maximum latency   : %.2f ms\n', result.max_latency * 1000);
    fprintf('Total sim time    : %.2f s\n', result.total_time);
end
