%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFile = '../experimentalDiffusionConstants/Diffusion Coefficients Munc13 Paper 1.xlsx';
sheetName = 'Fig2C&D Diff Co';

groupName = 'WT_BaCl2';

rawData = readExcelSheet(excelFile, sheetName);
rawDataVar = sprintf('rawData_%s', groupName);
assignin('base', rawDataVar, rawData);

categoryName = 'NO_STIM';

% Specify the range of column indices
startColumnIndex = 1;
endColumnIndex = 17;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 0;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categoryName = 'STIM';

% Specify the range of column indices
startColumnIndex = 20;
endColumnIndex = 36;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 3;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFile = '../experimentalDiffusionConstants/Diffusion Coefficients Munc13 Paper 1.xlsx';
sheetName = 'Fig7B';

groupName = 'Delta_C2A';

rawData = readExcelSheet(excelFile, sheetName);
rawDataVar = sprintf('rawData_%s', groupName);
assignin('base', rawDataVar, rawData);

categoryName = 'NO_STIM';

% Specify the range of column indices
startColumnIndex = 1;
endColumnIndex = 21;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 3;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categoryName = 'STIM';

% Specify the range of column indices
startColumnIndex = 24;
endColumnIndex = 44;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 3;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
excelFile = '../experimentalDiffusionConstants/Diffusion Coefficients Munc13 Paper 1.xlsx';
sheetName = 'Fig7E';

groupName = 'Delta_C2B';

rawData = readExcelSheet(excelFile, sheetName);
rawDataVar = sprintf('rawData_%s', groupName);
assignin('base', rawDataVar, rawData);

categoryName = 'NO_STIM';

% Specify the range of column indices
startColumnIndex = 1;
endColumnIndex = 18;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 3;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
categoryName = 'STIM';

% Specify the range of column indices
startColumnIndex = 21;
endColumnIndex = 38;

% Initialize an empty array to store the joined data
joinedDataVar = sprintf('%s_%s_Log10', groupName, categoryName);
joinedData = [];

for columnIndex = startColumnIndex:endColumnIndex
    skipLines = 3;
    
    result = extractColumnData(rawData, columnIndex, skipLines);
    
    resultVar = sprintf('%s_%s_%d', groupName, categoryName, columnIndex);
    
    assignin('base', resultVar, result);

    % Concatenate the result to the joinedData array
    joinedData = [joinedData; result];
end

assignin('base', joinedDataVar, joinedData);

dataVar = sprintf('%s_%s', groupName, categoryName);

assignin('base', dataVar, 10.^joinedData);