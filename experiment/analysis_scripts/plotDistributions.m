function [mean1, mean2] = plotDistributions(var1, var2, varargin)
% PLOTDISTRIBUTIONS Creates a publishable quality figure showing the distributions
% of two variables and their means
%
% Syntax:
%   [mean1, mean2] = plotDistributions(var1, var2)
%   [mean1, mean2] = plotDistributions(var1, var2, 'Name', Value, ...)
%
% Inputs:
%   var1 - First column of values
%   var2 - Second column of values
%
% Optional Name-Value Pairs:
%   'Title'      - Title for the plot (default: 'Distribution Comparison')
%   'XLabel'     - Label for x-axis (default: 'Value')
%   'YLabel'     - Label for y-axis (default: 'Frequency')
%   'Legend1'    - Legend for var1 (default: 'Variable 1')
%   'Legend2'    - Legend for var2 (default: 'Variable 2')
%   'BinMethod'  - Method for calculating bins (default: 'auto')
%                  Options: 'auto', 'scott', 'fd', 'sturges', or numeric value
%   'PlotType'   - Type of distribution plot (default: 'histogram')
%                  Options: 'histogram', 'ksdensity'
%   'Colors'     - 2x3 array of RGB values for colors (default: colorbrewer colors)
%   'SaveFig'    - Boolean to save figure (default: false)
%   'SavePath'   - Path to save figure (default: '../figures')
%   'FileName'   - Filename for saved figure (default: 'distribution_plot')
%   'FileFormat' - Format for saved figure (default: 'png')
%                  Options: 'png', 'pdf', 'eps', 'tiff', 'svg'
%   'Resolution' - Resolution in DPI for saved figure (default: 300)
%
% Outputs:
%   mean1      - Mean of var1
%   mean2      - Mean of var2
%
% Example:
%   x = randn(100, 1);
%   y = 2 + randn(100, 1);
%   [mx, my] = plotDistributions(x, y, 'Title', 'Normal Distributions', 'Legend1', 'Control', 'Legend2', 'Treatment');
%   
%   % To save the figure:
%   [mx, my] = plotDistributions(x, y, 'SaveFig', true, 'FileName', 'my_distribution');
%
% Copyright 2025

% Parse inputs
p = inputParser;
addRequired(p, 'var1', @isnumeric);
addRequired(p, 'var2', @isnumeric);
addParameter(p, 'Title', 'Distribution Comparison', @ischar);
addParameter(p, 'XLabel', 'Value', @ischar);
addParameter(p, 'YLabel', 'Frequency', @ischar);
addParameter(p, 'Legend1', 'Variable 1', @ischar);
addParameter(p, 'Legend2', 'Variable 2', @ischar);
addParameter(p, 'BinMethod', 'auto', @(x) ischar(x) || isnumeric(x));
addParameter(p, 'PlotType', 'histogram', @(x) ismember(x, {'histogram', 'ksdensity'}));
addParameter(p, 'Colors', [0.2157, 0.4941, 0.7216; 0.8941, 0.1020, 0.1098], @(x) ismatrix(x) && size(x,1) == 2 && size(x,2) == 3);
addParameter(p, 'SaveFig', false, @islogical);
addParameter(p, 'SavePath', '../figures', @ischar);
addParameter(p, 'FileName', 'distribution_plot', @ischar);
addParameter(p, 'FileFormat', 'png', @(x) ismember(x, {'png', 'pdf', 'eps', 'tiff', 'svg'}));
addParameter(p, 'Resolution', 300, @isnumeric);
parse(p, var1, var2, varargin{:});

opts = p.Results;

% Calculate means
mean1 = mean(var1);
mean2 = mean(var2);

% Create figure with nice size for publication (90mm is typical journal column width)
fig = figure;
set(fig, 'Units', 'centimeters', 'Position', [5, 5, 12, 9]);

% Setup for high-quality output
set(fig, 'Color', 'white');
set(fig, 'InvertHardcopy', 'off');

% Set default font properties for the figure
set(fig, 'DefaultAxesFontName', 'Arial');
set(fig, 'DefaultAxesFontSize', 9);
set(fig, 'DefaultTextFontName', 'Arial');
set(fig, 'DefaultTextFontSize', 9);

% Choose plot type
if strcmp(opts.PlotType, 'histogram')
    % Plot histograms
    h1 = histogram(var1, 'FaceColor', opts.Colors(1,:), 'FaceAlpha', 0.7, ...
        'EdgeColor', 'none', 'BinMethod', opts.BinMethod);
    hold on;
    h2 = histogram(var2, 'FaceColor', opts.Colors(2,:), 'FaceAlpha', 0.7, ...
        'EdgeColor', 'none', 'BinMethod', opts.BinMethod);
    
    % Plot mean lines
    % xLimits = xlim;
    % yLimits = ylim;
    % plot([mean1 mean1], [0 yLimits(2)*0.9], '--', 'Color', opts.Colors(1,:), 'LineWidth', 1.5);
    % plot([mean2 mean2], [0 yLimits(2)*0.9], '--', 'Color', opts.Colors(2,:), 'LineWidth', 1.5);
    
else % KDE plot
    [f1, xi1] = ksdensity(var1);
    [f2, xi2] = ksdensity(var2);
    
    plot(xi1, f1, 'Color', opts.Colors(1,:), 'LineWidth', 2);
    hold on;
    plot(xi2, f2, 'Color', opts.Colors(2,:), 'LineWidth', 2);
    
    % Fill areas under curves
    ax = gca;
    fill([xi1 fliplr(xi1)], [f1 zeros(size(f1))], opts.Colors(1,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    fill([xi2 fliplr(xi2)], [f2 zeros(size(f2))], opts.Colors(2,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    
    % Plot mean lines
    yLimits = ylim;
    plot([mean1 mean1], [0 yLimits(2)*0.9], '--', 'Color', opts.Colors(1,:), 'LineWidth', 1.5);
    plot([mean2 mean2], [0 yLimits(2)*0.9], '--', 'Color', opts.Colors(2,:), 'LineWidth', 1.5);
end

% Add text annotations for means
% text(mean1, yLimits(2)*0.95, ['\mu = ' num2str(mean1, '%.2f')], ...
%     'Color', opts.Colors(1,:), 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
% text(mean2, yLimits(2)*0.85, ['\mu = ' num2str(mean2, '%.2f')], ...
%     'Color', opts.Colors(2,:), 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% Add legend, title, and axis labels
legend({opts.Legend1, opts.Legend2}, 'Location', 'best', 'Box', 'off');
title(opts.Title, 'FontWeight', 'bold', 'FontSize', 11);
xlabel(opts.XLabel);
ylabel(opts.YLabel);
% set(gca, 'XScale', 'log');
xlim([0 0.5]);

% Fine-tune appearance for publication
box off;
set(gca, 'TickDir', 'out');
set(gca, 'LineWidth', 1);

% Add grid but make it subtle
% grid on;
% set(gca, 'GridAlpha', 0.15);
% set(gca, 'MinorGridAlpha', 0.05);

% Ensure tight margins and good layout
set(gca, 'Position', [0.13 0.15 0.80 0.75]);
set(gca, 'LooseInset', [0.05 0.05 0.05 0.05]);

% Save figure if requested
if opts.SaveFig
    % Create directory if it doesn't exist
    if ~exist(opts.SavePath, 'dir')
        mkdir(opts.SavePath);
        fprintf('Created directory: %s\n', opts.SavePath);
    end
    
    % Full path for saving
    fullPath = fullfile(opts.SavePath, [opts.FileName, '.', opts.FileFormat]);
    
    % Save with appropriate settings based on format
    switch opts.FileFormat
        case 'png'
            print(fig, fullPath, '-dpng', ['-r', num2str(opts.Resolution)]);
        case 'pdf'
            print(fig, fullPath, '-dpdf', ['-r', num2str(opts.Resolution)]);
        case 'eps'
            print(fig, fullPath, '-depsc', ['-r', num2str(opts.Resolution)]);
        case 'tiff'
            print(fig, fullPath, '-dtiff', ['-r', num2str(opts.Resolution)]);
        case 'svg'
            print(fig, fullPath, '-dsvg', ['-r', num2str(opts.Resolution)]);
    end
    
    fprintf('Figure saved to: %s\n', fullPath);
end

end