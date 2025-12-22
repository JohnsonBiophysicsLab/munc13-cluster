% function rawData = readExcelSheet(excelFile, sheetName)
%     % Read data from the specified Excel file and sheet
%     [~, ~, rawData] = xlsread(excelFile, sheetName);
% end
% xlsread only work for windows os with excel installed

function rawData = readExcelSheet(excelFile, sheetName)
    % Read data from the specified Excel file and sheet using readtable
    opts = detectImportOptions(excelFile, 'Sheet', sheetName);
    rawData = readtable(excelFile, opts);
end

