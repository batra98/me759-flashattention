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
let autoPlayInterval = null;
const captions = [
    "Press 'Next Step' to begin tile processing.",
    "Step 1: Load Q block (size Br × d) from HBM into SRAM.",
    "Step 2: Load K block (size Bc × d) from HBM into SRAM.",
    "Step 3: Compute S = Q × K^T matrix block. Update row max (m) in registers.",
    "Step 4: Load V block (size Bc × d) from HBM into SRAM.",
    "Step 5: Compute P = Softmax(S). Update O output block in registers.",
    "Step 6: Advance K & V to next block. (Flash loops Steps 2-5 without HBM writes)."
];

function toggleAutoPlay() {
    const btn = document.getElementById('autoPlayBtn');
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
        btn.innerText = '▶ Auto Play';
        btn.style.background = '';
    } else {
        tileStep(); // Take one step immediately
        autoPlayInterval = setInterval(tileStep, 1800);
        btn.innerText = '⏸ Pause';
        btn.style.background = '#f43f5e'; // Red to indicate it's active/can be stopped
    }
}

function tileStep() {
    animStep++;
    if (animStep > totalSteps) animStep = 1;
    updateAnimState();
}

function tileReset() {
    animStep = 0;
    if (autoPlayInterval) toggleAutoPlay();
    updateAnimState();
}

// Generate warp threads once on load
document.addEventListener('DOMContentLoaded', () => {
    const warpGrid = document.getElementById('warpGrid');
    if (warpGrid) {
        for (let i = 0; i < 32; i++) {
            let t = document.createElement('div');
            t.className = 'warp-thread';
            // Stagger animation delays for a wave effect when flashing
            t.style.animationDelay = `${(i % 8) * 0.02}s`;
            warpGrid.appendChild(t);
        }
    }
    
    // Generate bus traces
    const traces = document.getElementById('busTraces');
    if (traces) {
        for (let i = 0; i < 12; i++) {
            let tr = document.createElement('div');
            tr.className = 'trace';
            tr.style.animationDelay = `${Math.random() * 0.6}s`;
            traces.appendChild(tr);
        }
    }
});

function updateAnimState() {
    document.getElementById('stepCounter').innerText = `Step ${animStep} / ${totalSteps}`;
    document.getElementById('stepInfo').innerText = captions[animStep];

    const qSram = document.getElementById('sramQ');
    const kSram = document.getElementById('sramK');
    const vSram = document.getElementById('sramV');
    const dataBus = document.getElementById('dataBus');
    const busText = document.getElementById('busText');
    const computeRegs = document.getElementById('computeRegs');

    // Reset styles
    qSram.className = 'sram-tile'; qSram.innerHTML = '';
    kSram.className = 'sram-tile'; kSram.innerHTML = '';
    vSram.className = 'sram-tile'; vSram.innerHTML = '';
    qSram.style.transform = 'scale(1)'; kSram.style.transform = 'scale(1)'; vSram.style.transform = 'scale(1)';
    dataBus.classList.remove('active');
    computeRegs.classList.remove('compute-active');
    
    // Trigger CSS reflow for animation restart
    void computeRegs.offsetWidth;

    // Evaluate state
    if (animStep === 1 || animStep === 2 || animStep === 4) {
        dataBus.classList.add('active');
        busText.innerText = "COALESCED BURST: 320 GB/s";
    } else {
        busText.innerText = "MEMORY BUS IDLE";
    }

    if (animStep === 3 || animStep === 5) {
        computeRegs.classList.add('compute-active');
    }

    if (animStep >= 1) {
        qSram.classList.add('active-q');
        qSram.innerHTML = 'Q<sub>i</sub>';
        if (animStep === 1) qSram.style.transform = 'translateY(-10px) scale(1.1)';
    }
    if (animStep >= 2) {
        kSram.classList.add('active-k');
        kSram.innerHTML = 'K<sub>j</sub>';
        if (animStep === 2) kSram.style.transform = 'translateY(-10px) scale(1.1)';
    }
    if (animStep >= 4) {
        vSram.classList.add('active-v');
        vSram.innerHTML = 'V<sub>j</sub>';
        if (animStep === 4) vSram.style.transform = 'translateY(-10px) scale(1.1)';
    }

    const regLabel = document.getElementById('regLabel');
    if (animStep === 0) regLabel.innerHTML = '$m_i = -\\infty$, $\\ell_i = 0$, $O_i = 0$';
    if (animStep === 3) regLabel.innerHTML = 'Warp Compute: $S = QK^T$<br>Updating $m_i$...';
    if (animStep === 5) regLabel.innerHTML = 'Warp Compute: $P = Softmax(S)$<br>Updating $\\ell_i$, $O_i$...';
    
    // Re-render math if available
    if (window.renderMathInElement) {
        renderMathInElement(document.getElementById('regLabel'), {
            delimiters: [{left:'$', right:'$', display:false}]
        });
    }
}

// ============================================================================
// Intersection Observer (Scroll Reveal)
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log("Initializing scroll reveal...");
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.section > .container').forEach(el => {
        observer.observe(el);
    });
});
