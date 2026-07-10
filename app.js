// ────────────────────────────────────────────────────────────────
// MAIN APPLICATION - Core logic and UI updates
// ────────────────────────────────────────────────────────────────

// State management
const appState = {
    currentData: null,
    history: {
        temperature: [],
        humidity: [],
        gas: [],
        light: [],
        pressure: [],
        timestamps: []
    },
    lastHazard: 'SAFE',
    lastHazardKey: 'SAFE',
    isConnected: false,
    isUsingTestData: false,
    lastUpdateTime: null,
    backendBaseUrl: null
};

const cameraState = {
    stream: null,
    captureTimer: null,
    isInitializing: false
};

class AudioQueueManager {
    constructor() {
        this.queue = [];
        this.isPlaying = false;
        this.isUnlocked = false;
        this.lastEventId = null;
        this.audioFiles = {
            DANGER: 'fire.mp3',
            FIRE_RISK: 'fire.mp3',
            CONTROLLED_FIRE: 'c_f.mp3',
            GAS_LEAK: 'gas_leak.mp3',
            WATER_LEAK_RISK: 'water_leak.mp3',
            OVERHEATING: 'overheating.mp3',
            SMOKE_RISK: 'smoke.mp3'
        };
        this.audioCache = {};
        this.unlockAudio = this.unlockAudio.bind(this);
        this.preloadAudio();
    }

    preloadAudio() {
        Object.entries(this.audioFiles).forEach(([hazard, src]) => {
            if (this.audioCache[hazard]) return;

            const audio = new Audio(src);
            audio.preload = 'auto';
            audio.load();
            this.audioCache[hazard] = audio;
        });
    }

    initUnlockListeners() {
        ['click', 'touchstart', 'keydown'].forEach(eventName => {
            window.addEventListener(eventName, this.unlockAudio, { passive: true });
        });
    }

    async unlockAudio() {
        if (this.isUnlocked) return;

        try {
            for (const audio of Object.values(this.audioCache)) {
                try {
                    await this.primeAudioElement(audio);
                } catch (error) {
                    console.warn('[AUDIO] Prime failed:', error.message);
                }
            }

            this.isUnlocked = true;
            ['click', 'touchstart', 'keydown'].forEach(eventName => {
                window.removeEventListener(eventName, this.unlockAudio);
            });
            console.log('[AUDIO] Unlock Successful');
            this.processQueue();
        } catch (error) {
            console.warn('[AUDIO] Unlock blocked:', error.message);
        }
    }

    async primeAudioElement(audio) {
        audio.muted = true;
        audio.volume = 0;
        audio.currentTime = 0;

        try {
            await audio.play();
        } finally {
            audio.pause();
            audio.currentTime = 0;
            audio.muted = false;
            audio.volume = 1;
        }
    }

    enqueueFromBackend(newlyDetectedHazards = [], eventId = null) {
        const hazards = normalizeActiveHazards(newlyDetectedHazards, null);
        if (hazards.length === 0) return;

        if (eventId !== null && eventId === this.lastEventId) return;
        if (eventId !== null) this.lastEventId = eventId;

        hazards.forEach(hazard => {
            if (!this.audioFiles[hazard]) return;
            this.queue.push(hazard);
            console.log(`[AUDIO] Queued: ${hazard}`);
        });

        this.processQueue();
    }

    async processQueue() {
        if (this.isPlaying || !this.isUnlocked) return;

        this.isPlaying = true;
        while (this.queue.length > 0) {
            const hazard = this.queue.shift();
            await this.playHazardTwice(hazard);
        }
        this.isPlaying = false;
        console.log('[AUDIO] Queue Empty');
        if (this.queue.length > 0) this.processQueue();
    }

    async playHazardTwice(hazard) {
        for (let i = 0; i < 2; i++) {
            await this.playOnce(hazard);
        }
        console.log(`[AUDIO] Completed: ${hazard}`);
    }

    playOnce(hazard) {
        return new Promise(resolve => {
            const src = this.audioFiles[hazard];
            const audio = this.audioCache[hazard] || new Audio(src);
            let didFinish = false;
            let timeoutId = null;

            const finish = () => {
                if (didFinish) return;
                didFinish = true;
                clearTimeout(timeoutId);
                audio.removeEventListener('ended', finish);
                audio.removeEventListener('error', fail);
                resolve();
            };

            const fail = () => {
                console.warn(`[AUDIO] Playback Failed: ${hazard} (${src})`);
                finish();
            };

            audio.pause();
            audio.currentTime = 0;
            audio.addEventListener('ended', finish, { once: true });
            audio.addEventListener('error', fail, { once: true });
            timeoutId = setTimeout(finish, 15000);

            console.log(`[AUDIO] Playing: ${hazard}`);
            const playPromise = audio.play();
            if (playPromise && typeof playPromise.catch === 'function') {
                playPromise.catch(error => {
                    console.warn(`[AUDIO] Playback Failed: ${hazard} (${error.message})`);
                    finish();
                });
            }
        });
    }
}

const audioQueueManager = new AudioQueueManager();

// ────────────────────────────────────────────────────────────────
// FETCH DATA FROM BACKEND
// ────────────────────────────────────────────────────────────────

async function fetchFromBackend() {
    try {
        // Try multiple possible URLs
        const urls = [
            'http://localhost:5000/latest',
            'http://127.0.0.1:5000/latest',
            'http://10.89.93.70:5000/latest',
            // 'http//0.0.0.0:5000/latest'
             // Common local network IP
        ];

        let response = null;

        for (const url of urls) {
            try {
                response = await Promise.race([
                    fetch(url, { mode: 'cors', timeout: 2000 }),
                    new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('timeout')), 2000)
                    )
                ]);

                if (response && response.ok) {
                    console.log('✓ Connected to backend at:', url);
                    appState.backendBaseUrl = url.replace('/latest', '');
                    break;
                }
            } catch (e) {
                continue; // Try next URL
            }
        }

        if (!response || !response.ok) {
            throw new Error('Backend unreachable');
        }

        const data = await response.json();

        if (!data.current || !data.history) {
            throw new Error('Invalid data structure');
        }

        appState.currentData = data.current;
        appState.history = data.history;
        appState.isConnected = true;
        appState.isUsingTestData = false;

        console.log('Data updated:', data.current);
        return data;

    } catch (error) {
        console.warn('❌ Backend unavailable:', error.message);
        appState.isConnected = false;
        return null;
    }
}

// ────────────────────────────────────────────────────────────────
// USE TEST DATA (fallback)
// ────────────────────────────────────────────────────────────────

function useTestData() {
    if (!appState.isUsingTestData) {
        console.log('📊 Using test data (backend unavailable)');
        appState.isUsingTestData = true;
    }

    // Generate new reading
    const newReading = TestDataGenerator.generate();

    // Update current data
    appState.currentData = newReading;

    // Add to history (keep last 20)
    appState.history.temperature.push(newReading.temperature);
    appState.history.humidity.push(newReading.humidity);
    appState.history.gas.push(newReading.gas);
    appState.history.light.push(newReading.light);
    appState.history.pressure.push(newReading.pressure);
    appState.history.timestamps.push(newReading.timestamp);

    // Keep only last 20
    const maxLength = 20;
    if (appState.history.temperature.length > maxLength) {
        appState.history.temperature = appState.history.temperature.slice(-maxLength);
        appState.history.humidity = appState.history.humidity.slice(-maxLength);
        appState.history.gas = appState.history.gas.slice(-maxLength);
        appState.history.light = appState.history.light.slice(-maxLength);
        appState.history.pressure = appState.history.pressure.slice(-maxLength);
        appState.history.timestamps = appState.history.timestamps.slice(-maxLength);
    }

    return appState;
}

// ────────────────────────────────────────────────────────────────
// UPDATE UI
// ────────────────────────────────────────────────────────────────

function updateUI() {
    if (!appState.currentData) return;

    const data = appState.currentData;
    const history = appState.history;

    // Update temperature
    document.getElementById('temp-value').textContent = Formatter.temperature(data.temperature);
    const tempStatus = SensorStatus.determine('temperature', data.temperature);
    updateStatusBadge('temp-status', tempStatus);

    // Update humidity
    document.getElementById('humidity-value').textContent = Formatter.humidity(data.humidity);
    const humidStatus = SensorStatus.determine('humidity', data.humidity);
    updateStatusBadge('humidity-status', humidStatus);

    // Update gas
    document.getElementById('gas-value').textContent = Formatter.gas(data.gas);
    const gasStatus = SensorStatus.determine('gas', data.gas);
    updateStatusBadge('gas-status', gasStatus);

    // Update light
    document.getElementById('light-value').textContent = Formatter.light(data.light);
    const lightStatus = SensorStatus.determine('light', data.light);
    updateStatusBadge('light-status', lightStatus);

    // Update pressure
    document.getElementById('pressure-value').textContent = Formatter.pressure(data.pressure);
    const pressureStatus = SensorStatus.determine('pressure', data.pressure);
    updateStatusBadge('pressure-status', pressureStatus);

    // Update timestamp
    const timeStr = Formatter.timestamp(data.timestamp);
    document.getElementById('timestamp').textContent = `Last Update: ${timeStr}`;

    // Update connection status
    updateConnectionStatus();

    // Update hazard banner
    updateHazardBanner(data.active_hazards || [], data.hazard);
    audioQueueManager.enqueueFromBackend(
        data.newly_detected_hazards || [],
        data.hazard_event_id ?? data.hazard_event_timestamp ?? data.timestamp
    );

    // Update environment stability
    updateStability(history.pressure);

    // Draw charts
    if (history.temperature.length > 0) ChartDrawer.draw('temp-chart', history.temperature, '#ef4444');
    if (history.humidity.length > 0) ChartDrawer.draw('humidity-chart', history.humidity, '#06b6d4');
    if (history.gas.length > 0) ChartDrawer.draw('gas-chart', history.gas, '#fbbf24');
    if (history.light.length > 0) ChartDrawer.draw('light-chart', history.light, '#10b981');
    if (history.pressure.length > 0) ChartDrawer.draw('pressure-chart', history.pressure, '#a855f7');
}

// ────────────────────────────────────────────────────────────────
// UPDATE STATUS BADGE
// ────────────────────────────────────────────────────────────────

function updateStatusBadge(elementId, status) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.className = `status-badge ${status}`;

    const labels = {
        'safe': 'SAFE',
        'warning': 'WARNING',
        'danger': 'DANGER'
    };

    element.textContent = labels[status] || 'SAFE';
}

// ────────────────────────────────────────────────────────────────
// UPDATE HAZARD BANNER
// ────────────────────────────────────────────────────────────────

function normalizeActiveHazards(activeHazards, fallbackHazard) {
    if (Array.isArray(activeHazards) && activeHazards.length > 0) {
        return activeHazards.filter(hazard => hazard && hazard !== 'SAFE');
    }

    return fallbackHazard && fallbackHazard !== 'SAFE' ? [fallbackHazard] : [];
}

function updateHazardBanner(activeHazards = [], fallbackHazard = 'SAFE') {
    const banner = document.getElementById('hazard-banner');
    const icon = document.getElementById('hazard-icon');
    const text = document.getElementById('hazard-text');
    const hazards = normalizeActiveHazards(activeHazards, fallbackHazard);
    const primaryHazard = hazards[0] || 'SAFE';

    const hazardClass = HazardDetector.getHazardClass(primaryHazard);
    const hazardIcon = hazards.length > 1 ? '⚠' : HazardDetector.getHazardIcon(primaryHazard);
    const hazardText = HazardDetector.getHazardText(primaryHazard, hazards);

    banner.className = `hazard-banner ${hazardClass}`;
    icon.textContent = hazardIcon;
    text.innerHTML = hazardText;

    const hazardKey = hazards.length ? hazards.join('|') : 'SAFE';
    if (hazardKey !== appState.lastHazardKey) {
        if (hazards.length > 0) {
            showAlert(hazards);
        }
        appState.lastHazardKey = hazardKey;
        appState.lastHazard = primaryHazard;
    }
}

// ────────────────────────────────────────────────────────────────
// UPDATE STABILITY INDICATOR
// ────────────────────────────────────────────────────────────────

updateHazardBanner = function(activeHazards = [], fallbackHazard = 'SAFE') {
    const banner = document.getElementById('hazard-banner');
    const icon = document.getElementById('hazard-icon');
    const text = document.getElementById('hazard-text');
    const hazards = normalizeActiveHazards(activeHazards, fallbackHazard);
    const primaryHazard = hazards[0] || 'SAFE';

    banner.className = `hazard-banner ${HazardDetector.getHazardClass(primaryHazard)}`;
    icon.textContent = hazards.length > 1 ? 'MULTI' : HazardDetector.getHazardIcon(primaryHazard);
    text.innerHTML = HazardDetector.getHazardText(primaryHazard, hazards);

    const hazardKey = hazards.length ? hazards.join('|') : 'SAFE';
    if (hazardKey !== appState.lastHazardKey) {
        if (hazards.length > 0) {
            showAlert(hazards);
        }
        appState.lastHazardKey = hazardKey;
        appState.lastHazard = primaryHazard;
    }
};

function updateStability(pressureHistory) {
    const stability = StabilityCalculator.calculate(pressureHistory);

    const fillElement = document.getElementById('stability-fill');
    const valueElement = document.getElementById('stability-value');

    fillElement.style.width = `${stability.score}%`;
    fillElement.style.background = `linear-gradient(90deg, ${stability.color}88, ${stability.color})`;

    valueElement.textContent = `${stability.label} ${stability.score}%`;
    valueElement.style.color = stability.color;
}

// ────────────────────────────────────────────────────────────────
// UPDATE CONNECTION STATUS
// ────────────────────────────────────────────────────────────────

function updateConnectionStatus() {
    const statusText = document.getElementById('status-text');
    const dot = document.querySelector('.dot');

    if (appState.isConnected) {
        statusText.textContent = 'ONLINE - LIVE DATA';
        dot.className = 'dot online';
        document.getElementById('data-source').textContent = 'Data Source: ESP32 Hardware';
    } else if (appState.isUsingTestData) {
        statusText.textContent = 'OFFLINE - TEST MODE';
        dot.className = 'dot offline';
        document.getElementById('data-source').textContent = 'Data Source: Simulated Test Data';
    } else {
        statusText.textContent = 'CONNECTING...';
        dot.className = 'dot offline';
    }
}

// ────────────────────────────────────────────────────────────────
// SHOW ALERT
// ────────────────────────────────────────────────────────────────

function showAlert(hazardStatus) {
    const modal = document.getElementById('alert-modal');
    const title = document.getElementById('alert-title');
    const message = document.getElementById('alert-message');

    const titles = {
        'DANGER': '🔥 CRITICAL FIRE HAZARD!',
        'FIRE_RISK': '🔥 FIRE RISK DETECTED!',
        'CONTROLLED_FIRE': '⚠️ CONTROLLED FIRE WARNING',
        'WARNING': '⚠️ SYSTEM WARNING',
        'SMOKE_RISK': '⚠️ SMOKE RISK DETECTED!',
        'WATER_LEAK_RISK': '💧 WATER LEAK RISK!',
        'GAS_LEAK': '☠️ GAS LEAK DETECTED!',
        'OVERHEATING': '🌡️ OVERHEATING DETECTED!'
    };

    const messages = {
        'DANGER': 'Immediate action required! Fire hazard detected by sensors.',
        'FIRE_RISK': 'Deviation pattern matches rising heat/gas with fire-like air changes.',
        'CONTROLLED_FIRE': 'Controlled fire. Be cautious.',
        'WARNING': 'Environmental conditions are approaching dangerous levels.',
        'SMOKE_RISK': 'Deviation pattern matches gas rise, darker light readings, and humidity rise.',
        'WATER_LEAK_RISK': 'Deviation pattern matches sustained humidity and pressure rise.',
        'GAS_LEAK': 'Dangerous gas accumulation detected. Evacuate immediately!',
        'OVERHEATING': 'System temperature critically high. Check for fire sources!'
    };

    title.textContent = titles[hazardStatus] || 'HAZARD DETECTED';
    message.textContent = messages[hazardStatus] || 'System status has changed. Check details above.';

    modal.classList.remove('hidden');

    // Auto-close after 8 seconds if not dismissed
    setTimeout(() => {
        if (!modal.classList.contains('hidden')) {
            closeAlert();
        }
    }, 8000);
}

// ────────────────────────────────────────────────────────────────
// CLOSE ALERT
// ────────────────────────────────────────────────────────────────

showAlert = function(activeHazards) {
    const modal = document.getElementById('alert-modal');
    const title = document.getElementById('alert-title');
    const message = document.getElementById('alert-message');
    const hazards = normalizeActiveHazards(activeHazards, null);

    const titles = {
        'DANGER': 'CRITICAL FIRE HAZARD!',
        'FIRE_RISK': 'FIRE RISK DETECTED!',
        'CONTROLLED_FIRE': 'CONTROLLED FIRE WARNING',
        'WARNING': 'SYSTEM WARNING',
        'SMOKE_RISK': 'SMOKE RISK DETECTED!',
        'WATER_LEAK_RISK': 'WATER LEAK RISK!',
        'GAS_LEAK': 'GAS LEAK DETECTED!',
        'OVERHEATING': 'OVERHEATING DETECTED!'
    };

    const messages = {
        'DANGER': 'Immediate action required. Fire hazard detected by the fused sensor and camera system.',
        'FIRE_RISK': 'Fire-like environmental pattern detected.',
        'CONTROLLED_FIRE': 'Controlled fire detected by the CV model. Be cautious.',
        'WARNING': 'Environmental conditions are approaching dangerous levels.',
        'SMOKE_RISK': 'Smoke-like gas, light, or humidity pattern detected.',
        'WATER_LEAK_RISK': 'Sustained damp or leak-like pattern detected.',
        'GAS_LEAK': 'Dangerous gas accumulation detected. Evacuate immediately.',
        'OVERHEATING': 'Temperature pattern indicates possible overheating.'
    };

    if (hazards.length > 1) {
        title.textContent = 'MULTIPLE HAZARDS DETECTED';
        message.innerHTML = hazards
            .map(hazard => `<span class="alert-hazard-line">${HazardDetector.getHazardLabel(hazard)}</span>`)
            .join('');
    } else {
        const hazardStatus = hazards[0] || 'SAFE';
        title.textContent = titles[hazardStatus] || 'HAZARD DETECTED';
        message.textContent = messages[hazardStatus] || 'System status has changed. Check details above.';
    }

    modal.classList.remove('hidden');

    setTimeout(() => {
        if (!modal.classList.contains('hidden')) {
            closeAlert();
        }
    }, 8000);
};

function closeAlert() {
    const modal = document.getElementById('alert-modal');
    modal.classList.add('hidden');
}

// Camera capture
function isMobileDevice() {
    return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth <= 768;
}

function setCameraStatus(text, status = 'safe') {
    const badge = document.getElementById('camera-upload-status');
    if (!badge) return;

    badge.className = `status-badge ${status}`;
    badge.textContent = text;
}

function getCameraUploadUrls() {
    const urls = [];

    if (appState.backendBaseUrl) {
        urls.push(`${appState.backendBaseUrl}/camera-frame`);
    }

    if (window.location.hostname) {
        urls.push(`${window.location.protocol}//${window.location.hostname}:5000/camera-frame`);
    }

    urls.push('http://10.89.93.70:5000/camera-frame');

    return [...new Set(urls)];
}

function stopCameraCapture() {
    if (cameraState.captureTimer) {
        clearInterval(cameraState.captureTimer);
        cameraState.captureTimer = null;
    }

    if (cameraState.stream) {
        cameraState.stream.getTracks().forEach(track => track.stop());
        cameraState.stream = null;
    }

    const video = document.getElementById('camera-video');
    const fullscreenVideo = document.getElementById('camera-fullscreen-video');
    if (video) video.srcObject = null;
    if (fullscreenVideo) fullscreenVideo.srcObject = null;
}

async function initializeCamera() {
    const video = document.getElementById('camera-video');
    const fullscreenVideo = document.getElementById('camera-fullscreen-video');
    const placeholder = document.getElementById('camera-placeholder');

    if (!video || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraStatus('UNAVAILABLE', 'danger');
        if (placeholder) placeholder.textContent = 'CAMERA NOT AVAILABLE';
        return;
    }

    if (cameraState.stream || cameraState.isInitializing || document.hidden) return;

    try {
        cameraState.isInitializing = true;
        const constraints = {
            video: {
                facingMode: isMobileDevice() ? { exact: 'environment' } : { ideal: 'user' },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        try {
            cameraState.stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
            if (!isMobileDevice()) throw error;

            cameraState.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });
        }

        video.srcObject = cameraState.stream;
        if (fullscreenVideo) fullscreenVideo.srcObject = cameraState.stream;
        if (placeholder) placeholder.classList.add('hidden');

        setCameraStatus('LIVE', 'safe');
        startCameraCaptureLoop();
    } catch (error) {
        console.warn('Camera unavailable:', error.message);
        setCameraStatus('BLOCKED', 'danger');
        if (placeholder) placeholder.textContent = 'ALLOW CAMERA ACCESS';
    } finally {
        cameraState.isInitializing = false;
    }
}

function startCameraCaptureLoop() {
    if (cameraState.captureTimer) clearInterval(cameraState.captureTimer);

    captureAndUploadFrame();
    cameraState.captureTimer = setInterval(captureAndUploadFrame, 10000);
}

async function captureAndUploadFrame() {
    const video = document.getElementById('camera-video');
    if (document.hidden || !video || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const image = canvas.toDataURL('image/jpeg', 0.82);
    const payload = {
        image,
        captured_at: new Date().toISOString(),
        source: 'website-camera',
        device: isMobileDevice() ? 'mobile' : 'laptop'
    };

    for (const url of getCameraUploadUrls()) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'X-IDP-Camera-Source': 'website-camera'
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                setCameraStatus('SAVED', 'safe');
                return;
            }
        } catch (error) {
            continue;
        }
    }

    setCameraStatus('UPLOAD FAIL', 'warning');
}

function setupCameraFullscreen() {
    const card = document.getElementById('camera-card');
    const fullscreen = document.getElementById('camera-fullscreen');
    const backButton = document.getElementById('camera-back-btn');

    if (!card || !fullscreen || !backButton) return;

    card.addEventListener('click', () => {
        fullscreen.classList.remove('hidden');
        document.body.classList.add('camera-expanded');
    });

    backButton.addEventListener('click', (event) => {
        event.stopPropagation();
        fullscreen.classList.add('hidden');
        document.body.classList.remove('camera-expanded');
    });
}

// ────────────────────────────────────────────────────────────────
// MAIN UPDATE LOOP
// ────────────────────────────────────────────────────────────────

async function mainLoop() {
    // Try to fetch from backend
    const backendData = await fetchFromBackend();

    if (!backendData) {
        appState.isUsingTestData = false;
        updateConnectionStatus();
        document.getElementById('data-source').textContent = 'Data Source: Backend Offline';
        return;
    }

    // Update UI
    updateUI();
}

// ────────────────────────────────────────────────────────────────
// INITIALIZATION
// ────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    audioQueueManager.initUnlockListeners();
    console.log('🚀 IoT Hazard Monitor starting...');

    // Initial update
    mainLoop();

    // Set up polling (every 2 seconds)
    setInterval(mainLoop, 2000);

    setupCameraFullscreen();
    initializeCamera();

    console.log('✓ System ready. Polling every 2 seconds.');
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopCameraCapture();
        setCameraStatus('PAUSED', 'warning');
    } else {
        initializeCamera();
    }
});

window.addEventListener('beforeunload', stopCameraCapture);

// Allow manual alert close on click
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('alert-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeAlert();
            }
        });
    }
});
