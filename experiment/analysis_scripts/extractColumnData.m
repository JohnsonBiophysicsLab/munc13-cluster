function data = extractColumnData(rawData, columnIndex, skipLines)
    % Extract data from the specified column
    rawData = rawData(skipLines+1:end, :);
    data = rawData{:, columnIndex};
    
    % Remove non-numeric cells
    if iscell(data)
        data = data(cellfun(@isnumeric, data));
        % Convert cell array to numeric array
        data = cell2mat(data);
    end
    
    % Find the index of the last non-NaN value in data
    lastNonNanIndex = find(~isnan(data), 1, 'last');
    
    % Remove the NaN values at the end of data
    data = data(1:lastNonNanIndex);
end
