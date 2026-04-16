// ============================================================================
// UI Interactions — Code tabs
// ============================================================================
function showTab(tabId) {
    document.querySelectorAll('.code-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tabId}`).classList.add('active');
    
    // Find the button that called this and make it active
    const btns = document.querySelectorAll('.tab-btn');
    if (tabId === 'naive') btns[0].classList.add('active');
    if (tabId === 'flash') btns[1].classList.add('active');
}

// ============================================================================
// Tile Animation State Machine
// ============================================================================
let animStep = 0;
const totalSteps = 6;
const captions = [
    "Press 'Next Step' to begin tile processing.",
    "Step 1: Load Q block (size Br × d) from HBM into SRAM.",
    "Step 2: Load K block (size Bc × d) from HBM into SRAM.",
    "Step 3: Compute S = Q × K^T matrix block. Update row max (m) in registers.",
    "Step 4: Load V block (size Bc × d) from HBM into SRAM.",
    "Step 5: Compute P = Softmax(S). Update O output block in registers.",
    "Step 6: Advance K & V to next block. (Flash loops Steps 2-5 without HBM writes)."
];

function tileStep() {
    animStep++;
    if (animStep > totalSteps) animStep = 1;
    updateAnimState();
}

function tileReset() {
    animStep = 0;
    updateAnimState();
}

function updateAnimState() {
    document.getElementById('stepCounter').innerText = `Step ${animStep} / ${totalSteps}`;
    document.getElementById('stepInfo').innerText = captions[animStep];

    const qSram = document.getElementById('sramQ');
    const kSram = document.getElementById('sramK');
    const vSram = document.getElementById('sramV');

    // Reset styles
    qSram.style.background = 'transparent'; qSram.innerText = '';
    kSram.style.background = 'transparent'; kSram.innerText = '';
    vSram.style.background = 'transparent'; vSram.innerText = '';

    if (animStep >= 1) {
        qSram.style.background = 'rgba(244, 63, 94, 0.2)';
        qSram.style.border = '1px dashed #f43f5e';
        qSram.innerHTML = 'Q<sub>i</sub>';
    }
    if (animStep >= 2) {
        kSram.style.background = 'rgba(56, 189, 248, 0.2)';
        kSram.style.border = '1px dashed #38bdf8';
        kSram.innerHTML = 'K<sub>j</sub>';
    }
    if (animStep >= 4) {
        vSram.style.background = 'rgba(167, 139, 250, 0.2)';
        vSram.style.border = '1px dashed #a78bfa';
        vSram.innerHTML = 'V<sub>j</sub>';
    }

    const regLabel = document.getElementById('regLabel');
    if (animStep === 0) regLabel.innerHTML = '$m_i = -\\infty$, $\\ell_i = 0$, $O_i = 0$';
    if (animStep === 3) regLabel.innerHTML = 'Updating $m_i$...';
    if (animStep === 5) regLabel.innerHTML = 'Updating $\\ell_i$, $O_i$...';
    
    // Re-render math if available
    if (window.renderMathInElement) {
        renderMathInElement(document.getElementById('regLabel'), {
            delimiters: [{left:'$', right:'$', display:false}]
        });
    }
}

// ============================================================================
// CSV Parsing & Plotly Rendering
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Fetching benchmark data...');
    try {
        const [timingCsv, hbmCsv] = await Promise.all([
            fetch('data/timing.csv').then(r => r.text()),
            fetch('data/hbm_traffic.csv').then(r => r.text())
        ]);

        const timing = parseCSV(timingCsv);
        const hbm = parseCSV(hbmCsv);
        
        plotTiming(timing);
        plotHBM(hbm);

        // Update hero metric dynamically if valid data
        const n8192_naive_timing = timing.find(d => d.mode === 'naive' && d.seq_len == '8192');
        const n8192_flash_timing = timing.find(d => d.mode === 'flash' && d.seq_len == '8192');
        
        const n8192_naive_hbm = hbm.find(d => d.mode === 'naive' && d.seq_len == '8192');
        const n8192_flash_hbm = hbm.find(d => d.mode === 'flash' && d.seq_len == '8192');

        if (n8192_naive_timing && n8192_flash_timing) {
            const speedup = (n8192_naive_timing.ms / n8192_flash_timing.ms).toFixed(1);
            const speedupEl = document.getElementById('speedupLabel');
            if (speedupEl) speedupEl.innerText = `~${speedup}×`;
        }

        if (n8192_naive_hbm && n8192_flash_hbm) {
            const naive_hbm = parseFloat(n8192_naive_hbm.bytes_read_MB) + parseFloat(n8192_naive_hbm.bytes_write_MB);
            const flash_hbm = parseFloat(n8192_flash_hbm.bytes_read_MB) + parseFloat(n8192_flash_hbm.bytes_write_MB);
            const hbmRatio = (naive_hbm / flash_hbm).toFixed(0);
            const hbmEl = document.getElementById('hbmLabel');
            if (hbmEl) hbmEl.innerText = `~${hbmRatio}×`;
        }

    } catch (e) {
        console.error("Failed to load CSV data. If testing locally, launch a local web server (e.g. python -m http.server)");
        document.getElementById('chartTiming').innerText = "Loading data failed. Are CSV files available at /data/?";
        document.getElementById('chartHBM').innerText = "Loading data failed. Are CSV files available at /data/?";
    }
});

function parseCSV(str) {
    const lines = str.trim().split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        const row = line.split(',');
        const obj = {};
        headers.forEach((h, i) => obj[h.trim()] = row[i].trim());
        return obj;
    });
}

const theme = {
    font: { family: 'Inter, sans-serif', color: '#e2e4ef' },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    gridcolor: '#2a2d3d'
};

function plotTiming(data) {
    const naive = data.filter(d => d.mode === 'naive');
    const flash = data.filter(d => d.mode === 'flash');

    const trace1 = {
        x: naive.map(d => d.seq_len),
        y: naive.map(d => d.ms),
        name: 'Naive Attention',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#f43f5e', width: 3 },
        marker: { size: 8 }
    };

    const trace2 = {
        x: flash.map(d => d.seq_len),
        y: flash.map(d => d.ms),
        name: 'FlashAttention',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#38bdf8', width: 3 },
        marker: { size: 8 }
    };

    const layout = {
        ...theme,
        margin: { l: 50, r: 20, t: 30, b: 50 },
        xaxis: { title: 'Sequence Length (N)', type: 'category', gridcolor: theme.gridcolor },
        yaxis: { title: 'Latency (ms)', gridcolor: theme.gridcolor },
        legend: { x: 0.05, y: 0.95, bgcolor: 'rgba(15,16,26,0.8)' }
    };

    Plotly.newPlot('chartTiming', [trace1, trace2], layout, {responsive: true});
}

function plotHBM(data) {
    // We sum read and write traffic
    const naive = data.filter(d => d.mode === 'naive');
    const flash = data.filter(d => d.mode === 'flash');

    const trace1 = {
        x: naive.map(d => d.seq_len),
        y: naive.map(d => parseFloat(d.bytes_read_MB) + parseFloat(d.bytes_write_MB)),
        name: 'Naive HBM Traffic',
        type: 'bar',
        marker: { color: '#f43f5e' }
    };

    const trace2 = {
        x: flash.map(d => d.seq_len),
        y: flash.map(d => parseFloat(d.bytes_read_MB) + parseFloat(d.bytes_write_MB)),
        name: 'Flash HBM Traffic',
        type: 'bar',
        marker: { color: '#38bdf8' }
    };

    const layout = {
        ...theme,
        barmode: 'group',
        margin: { l: 60, r: 20, t: 30, b: 50 },
        xaxis: { title: 'Sequence Length (N)', type: 'category', gridcolor: theme.gridcolor },
        yaxis: { title: 'Total HBM Traffic (MB)', type: 'log', gridcolor: theme.gridcolor },
        legend: { x: 0.05, y: 0.95, bgcolor: 'rgba(15,16,26,0.8)' }
    };

    Plotly.newPlot('chartHBM', [trace1, trace2], layout, {responsive: true});
}
