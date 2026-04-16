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
