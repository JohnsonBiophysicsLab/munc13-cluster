%% Effect of Dimerization on Unc13A Mobility and Clustering
% This script generates publication-quality figures comparing 
% wild-type and ΔC2A mutant Unc13A dynamics

% Clear workspace and set up figure defaults
% clear all;
% close all;
% clc;

% Set figure defaults for publication quality
set(0, 'DefaultFigureColor', 'white');
set(0, 'DefaultAxesFontSize', 20);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultAxesLineWidth', 1.5);
set(0, 'DefaultLineLineWidth', 2);

% Define custom colors
wt_color = [0.00, 0.45, 0.74];       % Blue for wild-type
mutant_color = [0.85, 0.33, 0.10];   % Red for ΔC2A mutant
nostim_color = [0.47, 0.67, 0.19];   % Green for NO STIM
stim_color = [0.49, 0.18, 0.56];     % Purple for STIM

%% Load CSV data

% Define file paths
wt_file = '../odeSolution/wt_cluster_munc13_copies.csv';
deltac2a_file = '../odeSolution/deltac2a_cluster_munc13_copies.csv';

% Load the data using readtable
wt_data = readtable(wt_file);
deltac2a_data = readtable(deltac2a_file);

%% Figure: Time Series Analysis of Unc13A Clustering
try
    % Check if we successfully loaded the data
    if exist('wt_data', 'var') && exist('deltac2a_data', 'var')
        % Create figure for time series
        figure('Position', [100, 100, 1200, 800]);
        
        % Subplot 1: Unc13A copies in cluster over time
        subplot(2, 1, 1);
        
        % Plot time series
        plot(wt_data.timePoints, wt_data.preCluster, 'b--', 'LineWidth', 2);
        hold on;
        plot(wt_data.timePoints, wt_data.postCluster, 'b-', 'LineWidth', 2);
        plot(deltac2a_data.timePoints, deltac2a_data.preCluster, 'r--', 'LineWidth', 2);
        plot(deltac2a_data.timePoints, deltac2a_data.postCluster, 'r-', 'LineWidth', 2);
        
        % Customize plot
        xlabel('Time (s)');
        ylabel('Unc13A Copies in Cluster');
        % title('Temporal Dynamics of Unc13A Recruitment to Clusters', 'FontWeight', 'normal');
        % legend({'WT NO STIM', 'WT STIM', 'ΔC2A NO STIM', 'ΔC2A STIM'}, 'Location', 'southeast');
        % grid on;
        
        % Subplot 2: Percentage in cluster over time
        subplot(2, 1, 2);
        
        % Calculate percentages over time
        wt_no_stim_pct = (wt_data.preCluster ./ wt_data.pre) * 100;
        wt_stim_pct = (wt_data.postCluster ./ wt_data.post) * 100;
        deltac2a_no_stim_pct = (deltac2a_data.preCluster ./ deltac2a_data.pre) * 100;
        deltac2a_stim_pct = (deltac2a_data.postCluster ./ deltac2a_data.post) * 100;
        
        % Plot time series
        plot(wt_data.timePoints, wt_no_stim_pct, 'b--', 'LineWidth', 2);
        hold on;
        plot(wt_data.timePoints, wt_stim_pct, 'b-', 'LineWidth', 2);
        plot(deltac2a_data.timePoints, deltac2a_no_stim_pct, 'r--', 'LineWidth', 2);
        plot(deltac2a_data.timePoints, deltac2a_stim_pct, 'r-', 'LineWidth', 2);
        
        % Customize plot
        xlabel('Time (s)');
        ylabel('Unc13A in Clusters (%)');
        % title('Temporal Dynamics of Relative Unc13A Enrichment', 'FontWeight', 'normal');
        legend({'WT NO STIM', 'WT STIM', 'ΔC2A NO STIM', 'ΔC2A STIM'}, 'Location', 'southeast');
        % grid on;
        
        % Calculate the delta between STIM and NO STIM for each genotype
        % annotation('textbox', [0.15, 0.1, 0.7, 0.1], 'String', ...
        %     {'WT: Stimulation increases cluster enrichment', ...
        %      'ΔC2A: Stimulation decreases cluster enrichment', ...
        %      'Conclusion: Dimerization is required for activity-dependent recruitment of Unc13A to release sites'}, ...
        %     'FitBoxToText', 'on', 'HorizontalAlignment', 'center', ...
        %     'FontSize', 12, 'EdgeColor', 'none');
    end
end

print('../figures/Fig2_DimerizationEffect_Dynamics', '-dpng', '-r300');
print('../figures/Fig2_DimerizationEffect_Dynamics', '-dsvg');
