// One ordered save queue per editor. Capture its page and value rather than
// reading whichever editor or page happens to be active when a save completes.
function createTextAutosave({initialValue, read, write, report, delay = 500}) {
    let saved = initialValue;
    let timer = null;
    let pending = null;
    let running = null;
    let disposed = false;

    async function drain() {
        while (pending !== null) {
            const value = pending;
            pending = null;
            if (value === saved) continue;
            report('Saving…', false);
            try {
                const result = await write(value);
                if (!result || result.success === false) throw new Error(result?.error || 'Save failed');
                saved = value;
                report('Saved', false);
            } catch (error) {
                report('Error: ' + error.message, true);
                // Keep the edit available for a manual retry.
                if (pending === null) break;
            }
        }
    }

    function flush() {
        clearTimeout(timer);
        timer = null;
        pending = read();
        if (!running) {
            running = drain().finally(() => {
                running = null;
                if (pending !== null) flush();
            });
        }
        return running;
    }

    return {
        changed() {
            if (disposed) return;
            clearTimeout(timer);
            timer = setTimeout(flush, delay);
        },
        flush,
        dispose() {
            if (disposed) return running || Promise.resolve();
            disposed = true;
            return flush();
        },
    };
}

if (typeof module !== 'undefined' && module.exports) module.exports = {createTextAutosave};
