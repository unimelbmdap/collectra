// The import summary, then a line for each file that failed
function importNoticeText(info) {
    if (!info || !info.summary) return '';
    const failed = (info.failed || []).map((entry) => `Failed: ${entry}`);
    return [info.summary, ...failed].join('\n');
}

if (typeof module !== 'undefined') module.exports = {importNoticeText};
