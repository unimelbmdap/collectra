// What the page says after opening imported raw inputs: the summary, then
// one line for each file that could not be imported.
function importNoticeText(info) {
    if (!info || !info.summary) return '';
    const failed = (info.failed || []).map((entry) => `Failed: ${entry}`);
    return [info.summary, ...failed].join('\n');
}

if (typeof module !== 'undefined') module.exports = {importNoticeText};
