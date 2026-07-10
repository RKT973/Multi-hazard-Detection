const historyState = {
    backendBaseUrl: null
};

function getBackendCandidates(path) {
    const urls = [];

    if (historyState.backendBaseUrl) {
        urls.push(`${historyState.backendBaseUrl}${path}`);
    }

    if (window.location.hostname) {
        urls.push(`${window.location.protocol}//${window.location.hostname}:5000${path}`);
    }

    urls.push(`http://10.89.93.70:5000${path}`);
    urls.push(`http://localhost:5000${path}`);

    return [...new Set(urls)];
}

async function fetchHazardHistory() {
    for (const url of getBackendCandidates('/hazard-history')) {
        try {
            const response = await fetch(url, { mode: 'cors' });
            if (!response.ok) continue;

            historyState.backendBaseUrl = url.replace('/hazard-history', '');
            return await response.json();
        } catch (error) {
            continue;
        }
    }

    throw new Error('Backend unreachable');
}

function formatDateTime(value) {
    if (!value) return '--';

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '--';

    return date.toLocaleString([], {
        year: 'numeric',
        month: 'short',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function renderHistory(entries) {
    const list = document.getElementById('history-list');
    const status = document.getElementById('history-status');

    if (!entries || entries.length === 0) {
        list.innerHTML = '<div class="history-empty">No hazard events recorded yet.</div>';
        status.className = 'status-badge safe';
        status.textContent = 'EMPTY';
        return;
    }

    status.className = 'status-badge safe';
    status.textContent = `${entries.length} EVENTS`;

    list.innerHTML = entries.map(entry => {
        const hazardClass = HazardDetector.getHazardClass(entry.hazard);
        const hazardLabel = HazardDetector.getHazardLabel(entry.hazard);
        const lifecycle = entry.resolved ? 'RESOLVED' : 'ACTIVE';
        const lifecycleClass = entry.resolved ? 'resolved' : 'active';
        const severity = (entry.status || 'warning').toUpperCase();
        const resolvedText = entry.resolved
            ? `<div><span>Resolved</span>${formatDateTime(entry.resolved_at)}</div>`
            : '<div><span>Resolved</span>Pending</div>';

        return `
            <article class="history-item ${hazardClass}">
                <div class="history-item-main">
                    <div class="history-hazard">${hazardLabel}</div>
                    <div class="history-meta">
                        <div><span>Detected</span>${formatDateTime(entry.detected_at)}</div>
                        ${resolvedText}
                    </div>
                </div>
                <div class="history-badges">
                    <span class="history-status-pill ${lifecycleClass}">${lifecycle}</span>
                    <span class="history-severity-pill ${entry.status || 'warning'}">${severity}</span>
                </div>
            </article>
        `;
    }).join('');
}

async function updateHistoryPage() {
    const status = document.getElementById('history-status');

    try {
        const data = await fetchHazardHistory();
        renderHistory(data.history || []);
    } catch (error) {
        status.className = 'status-badge warning';
        status.textContent = 'OFFLINE';
        document.getElementById('history-list').innerHTML =
            '<div class="history-empty">Unable to reach backend.</div>';
    }
}

window.addEventListener('DOMContentLoaded', () => {
    updateHistoryPage();
    setInterval(updateHistoryPage, 5000);
});
