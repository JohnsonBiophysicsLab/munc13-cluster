%% Unc13A Diffusion Analysis and Visualization
% This script generates publication-quality figures to illustrate the
% changes in Unc13A diffusion after stimulation

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
blue_color = [0.00, 0.45, 0.74];
red_color = [0.85, 0.33, 0.10];
green_color = [0.47, 0.67, 0.19];
purple_color = [0.49, 0.18, 0.56];
orange_color = [0.93, 0.69, 0.13];

%% Figure 1: Experimental vs. Model Diffusion Constants
figure('Position', [100, 100, 500, 500]);

% Data for diffusion constants
conditions = {'NO STIM', 'STIM'};
exp_diffusion = [0.1085, 0.0745]; % Experimental values (μm²/s)
model_diffusion = [0.1019, 0.0810]; % Model predictions (μm²/s)

% Create grouped bar plot
bar_h = bar([exp_diffusion', model_diffusion']);
set(bar_h(1), 'FaceColor', blue_color);
set(bar_h(2), 'FaceColor', red_color);

% Add error bars (SEM)
% hold on;
% errorbar([0.85, 1.85], exp_diffusion, exp_diffusion*0.05, 'k.', 'LineWidth', 1.5);
% errorbar([1.15, 2.15], model_diffusion, model_diffusion*0.05, 'k.', 'LineWidth', 1.5);
% hold off;

% Customize plot appearance
set(gca, 'XTickLabel', conditions);
ylabel('Diffusion Constant (μm²/s)');
ylim([0, 0.15]);
% title('Unc13A Diffusion: Experimental vs. Model', 'FontWeight', 'normal');
legend({'Experimental', 'Model'}, 'Location', 'northeast');
% box on;

% Add p-value for experimental data
% pval_str = 'p < 0.001';
% text(1.5, 0.14, pval_str, 'HorizontalAlignment', 'center', 'FontSize', 12);

% Add bracket for p-value
% annotation('line', [0.35, 0.65], [0.92, 0.92], 'LineWidth', 1.5);
% annotation('line', [0.35, 0.35], [0.90, 0.92], 'LineWidth', 1.5);
% annotation('line', [0.65, 0.65], [0.90, 0.92], 'LineWidth', 1.5);
print('../figures/Fig1_DiffusionConstants', '-dpng', '-r300');
print('../figures/Fig1_DiffusionConstants', '-dsvg');

%% Figure 2: Molecular State Distribution
figure('Position', [100, 100, 500, 500]);

% Data for molecular states
states = {'Monomer (MR)', 'Dimer (MMR)', 'Dimer (MMRR)'};
no_stim_states = [45.1, 12.6, 42.3]; % NO STIM percentages
stim_states = [29.5, 5.1, 65.4]; % STIM percentages

% Create grouped bar plot
bar_h = bar([no_stim_states', stim_states']);
set(bar_h(1), 'FaceColor', blue_color);
set(bar_h(2), 'FaceColor', red_color);

% Customize plot appearance
set(gca, 'XTickLabel', states);
ylabel('Percentage (%)');
ylim([0, 80]);
% title('Unc13A Molecular State Distribution', 'FontWeight', 'normal');
legend({'NO STIM', 'STIM'}, 'Location', 'northwest');
% box on;

% Add percentage labels on bars
xtips1 = bar_h(1).XEndPoints;
ytips1 = bar_h(1).YEndPoints;
labels1 = string(bar_h(1).YData) + '%';
text(xtips1, ytips1, labels1, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 12);

xtips2 = bar_h(2).XEndPoints;
ytips2 = bar_h(2).YEndPoints;
labels2 = string(bar_h(2).YData) + '%';
text(xtips2, ytips2, labels2, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 12);
print('../figures/Fig2_MolecularStates', '-dpng', '-r300');
print('../figures/Fig2_MolecularStates', '-dsvg');

%% Figure 3: Cluster Confinement and Membrane Diffusion
figure('Position', [100, 100, 1000, 500]);

% Create subplot for cluster confinement
subplot(1, 2, 1);
cluster_data = [32.27, 36.95]; % Percentage in clusters

bar_h = bar(cluster_data);
set(bar_h, 'FaceColor', green_color);

% Customize plot appearance
set(gca, 'XTickLabel', conditions);
ylabel('Percentage in Clusters (%)');
ylim([0, 45]);
% title('Unc13A Cluster Confinement', 'FontWeight', 'normal');
% box on;

% Add percentage labels on bars
xtips = bar_h.XEndPoints;
ytips = bar_h.YEndPoints;
labels = string(bar_h.YData) + '%';
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 12);

% Create subplot for membrane diffusion
subplot(1, 2, 2);
membrane_diffusion = [0.1504, 0.1285]; % Membrane diffusion constants

bar_h = bar(membrane_diffusion);
set(bar_h, 'FaceColor', purple_color);

% Customize plot appearance
set(gca, 'XTickLabel', conditions);
ylabel('Diffusion Constant (μm²/s)');
ylim([0, 0.2]);
% title('Unc13A Membrane Diffusion', 'FontWeight', 'normal');
% box on;

% Add value labels on bars
xtips = bar_h.XEndPoints;
ytips = bar_h.YEndPoints;
labels = string(round(bar_h.YData, 4));
text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', 12);
print('../figures/Fig3_ClusterAndMembrane', '-dpng', '-r300');
print('../figures/Fig3_ClusterAndMembrane', '-dsvg');
