% run analuzeExperimentData.m first

% plot
[mx, my] = plotDistributions(WT_BaCl2_NO_STIM, WT_BaCl2_STIM, 'Title', 'WT', 'Legend1', 'NO STIM', 'Legend2', 'STIM', 'SaveFig', true, 'FileName', 'figS1')