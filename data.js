// ────────────────────────────────────────────────────────────────
// DATA UTILITIES - Chart drawing and data management
// ────────────────────────────────────────────────────────────────

class ChartDrawer {
    static draw(canvasId, data, color) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !data || data.length === 0) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        if (data.length < 2) return;

        // Calculate bounds
        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const range = max - min || 1;
        const step = width / (data.length - 1);

        // Draw line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        data.forEach((value, index) => {
            const x = index * step;
            const y = height - ((value - min) / range) * height;

            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Draw fill gradient
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();

        // Create gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, color + '66');
        gradient.addColorStop(1, color + '00');
        ctx.fillStyle = gradient;
        ctx.fill();
    }
}

// ────────────────────────────────────────────────────────────────
// HAZARD STATUS DETERMINATION
// ────────────────────────────────────────────────────────────────

class HazardDetector {
    static getHazardClass(hazardStatus) {
        // Map hazard strings to CSS class names
        const mapping = {
            'DANGER': 'danger',
            'FIRE_RISK': 'danger',
            'FIRE_RISK': 'danger',
            'FIRE_RISK': 'danger',
            'WARNING': 'warning',
            'SMOKE_RISK': 'warning',
            'WATER_LEAK_RISK': 'water_leak',
            'SMOKE_RISK': 'warning',
            'WATER_LEAK_RISK': 'water_leak',
            'SMOKE_RISK': 'warning',
            'WATER_LEAK_RISK': 'water_leak',
            'GAS_LEAK': 'gas_leak',
            'OVERHEATING': 'overheating',
            'SAFE': 'safe'
        };
        return mapping[hazardStatus] || 'safe';
    }

    static getHazardIcon(hazardStatus) {
        const mapping = {
            'DANGER': '🔥',
            'FIRE_RISK': '🔥',
            'FIRE_RISK': '🔥',
            'FIRE_RISK': '🔥',
            'WARNING': '⚠️',
            'SMOKE_RISK': '⚠️',
            'WATER_LEAK_RISK': '💧',
            'SMOKE_RISK': '⚠️',
            'WATER_LEAK_RISK': '💧',
            'SMOKE_RISK': '⚠️',
            'WATER_LEAK_RISK': '💧',
            'GAS_LEAK': '☠️',
            'OVERHEATING': '🌡️',
            'SAFE': '✓'
        };
        return mapping[hazardStatus] || '✓';
    }

    static getHazardText(hazardStatus) {
        return `SYSTEM: ${hazardStatus}`;
    }
}

// ────────────────────────────────────────────────────────────────
// STABILITY CALCULATION
// ────────────────────────────────────────────────────────────────

class StabilityCalculator {
    static calculate(pressureHistory) {
        if (!pressureHistory || pressureHistory.length < 2) {
            return {
                score: 100,
                label: 'STABLE',
                color: '#10b981'
            };
        }

        const mean = pressureHistory.reduce((a, b) => a + b, 0) / pressureHistory.length;
        const variance = pressureHistory.reduce(
            (a, b) => a + Math.pow(b - mean, 2), 
            0
        ) / pressureHistory.length;
        const stdDev = Math.sqrt(variance);

        // Map stdDev: 0→100% stable, 5+ hPa→0% stable
        const score = Math.max(0, Math.min(100, 100 - (stdDev / 5) * 100));
        const rounded = Math.round(score);

        let label, color;
        if (rounded > 70) {
            label = 'STABLE';
            color = '#10b981';
        } else if (rounded > 40) {
            label = 'VARIABLE';
            color = '#fbbf24';
        } else {
            label = 'UNSTABLE';
            color = '#ef4444';
        }

        return { score: rounded, label, color };
    }
}

// ────────────────────────────────────────────────────────────────
// SENSOR STATUS DETERMINATION
// ────────────────────────────────────────────────────────────────

class SensorStatus {
    static determine(sensorType, value) {
        // Return status based on sensor type and value
        const thresholds = {
            temperature: { warning: 35, danger: 40 },
            humidity: { warning: 70, danger: 80 },
            gas: { warning: 400, danger: 700 },
            light: { warning: 100, danger: 50 },
            pressure: { warning: 950, danger: 1000 } // Rough thresholds
        };

        const thresh = thresholds[sensorType];
        if (!thresh) return 'safe';

        if (value >= thresh.danger) return 'danger';
        if (value >= thresh.warning) return 'warning';
        return 'safe';
    }
}

// ────────────────────────────────────────────────────────────────
// SAMPLE DATA GENERATOR (for testing without backend)
// ────────────────────────────────────────────────────────────────

class TestDataGenerator {
    static generate() {
        const baseTemp = 24.5;
        const baseHumidity = 55;
        const baseGas = 350;
        const baseLight = 650;
        const basePressure = 1013.25;

        return {
            temperature: baseTemp + (Math.random() - 0.5) * 3,
            humidity: baseHumidity + (Math.random() - 0.5) * 10,
            gas: baseGas + (Math.random() - 0.5) * 50,
            light: baseLight + (Math.random() - 0.5) * 100,
            pressure: basePressure + (Math.random() - 0.5) * 2,
            hazard: Math.random() > 0.95 ? 'WARNING' : 'SAFE',
            timestamp: new Date().toISOString()
        };
    }

    static generateHistory(count = 5) {
        const history = [];
        for (let i = 0; i < count; i++) {
            history.push(this.generate());
        }
        return history;
    }

    static formatForDisplay(rawHistory) {
        // Convert array of sensor readings to organized format
        const formatted = {
            temperature: [],
            humidity: [],
            gas: [],
            light: [],
            pressure: [],
            timestamps: []
        };

        rawHistory.forEach(reading => {
            formatted.temperature.push(reading.temperature);
            formatted.humidity.push(reading.humidity);
            formatted.gas.push(reading.gas);
            formatted.light.push(reading.light);
            formatted.pressure.push(reading.pressure);
            formatted.timestamps.push(reading.timestamp);
        });

        return formatted;
    }
}

// ────────────────────────────────────────────────────────────────
// FORMAT UTILITIES
// ────────────────────────────────────────────────────────────────

class Formatter {
    static timestamp(isoString) {
        if (!isoString) return '--:--:--';
        const date = new Date(isoString);
        return date.toLocaleTimeString();
    }

    static temperature(value) {
        return isNaN(value) ? '--' : value.toFixed(1);
    }

    static humidity(value) {
        return isNaN(value) ? '--' : value.toFixed(1);
    }

    static gas(value) {
        return isNaN(value) ? '--' : Math.round(value);
    }

    static light(value) {
        return isNaN(value) ? '--' : Math.round(value);
    }

    static pressure(value) {
        return isNaN(value) ? '--' : value.toFixed(1);
    }
}

// ────────────────────────────────────────────────────────────────
// EXPORT for use in app.js
// ────────────────────────────────────────────────────────────────

// Global utilities (accessed from app.js)
window.ChartDrawer = ChartDrawer;
window.HazardDetector = HazardDetector;
window.StabilityCalculator = StabilityCalculator;
window.SensorStatus = SensorStatus;
window.TestDataGenerator = TestDataGenerator;
window.Formatter = Formatter;

HazardDetector.getHazardClass = function(hazardStatus) {
    const mapping = {
        'DANGER': 'danger',
        'FIRE_RISK': 'danger',
        'CONTROLLED_FIRE': 'warning',
        'WARNING': 'warning',
        'SMOKE_RISK': 'warning',
        'WATER_LEAK_RISK': 'water_leak',
        'GAS_LEAK': 'gas_leak',
        'OVERHEATING': 'overheating',
        'SAFE': 'safe'
    };
    return mapping[hazardStatus] || 'safe';
};

HazardDetector.getHazardIcon = function(hazardStatus) {
    const mapping = {
        'DANGER': 'FIRE',
        'FIRE_RISK': 'FIRE',
        'CONTROLLED_FIRE': 'FIRE',
        'WARNING': 'WARN',
        'SMOKE_RISK': 'SMOKE',
        'WATER_LEAK_RISK': 'WATER',
        'GAS_LEAK': 'GAS',
        'OVERHEATING': 'HEAT',
        'SAFE': 'OK'
    };
    return mapping[hazardStatus] || 'OK';
};

HazardDetector.getHazardLabel = function(hazardStatus) {
    const mapping = {
        'DANGER': 'FIRE DANGER',
        'FIRE_RISK': 'FIRE RISK',
        'CONTROLLED_FIRE': 'CONTROLLED FIRE',
        'WARNING': 'SYSTEM WARNING',
        'SMOKE_RISK': 'SMOKE',
        'WATER_LEAK_RISK': 'WATER LEAK',
        'GAS_LEAK': 'GAS LEAK',
        'OVERHEATING': 'OVERHEATING',
        'SAFE': 'SAFE'
    };
    return mapping[hazardStatus] || hazardStatus;
};

HazardDetector.getHazardText = function(hazardStatus, activeHazards = []) {
    if (!activeHazards || activeHazards.length === 0 || hazardStatus === 'SAFE') {
        return 'SYSTEM: SAFE';
    }

    if (activeHazards.length === 1) {
        return this.getHazardLabel(hazardStatus);
    }

    return activeHazards
        .map(hazard => `<span>${this.getHazardLabel(hazard)}</span>`)
        .join('');
};
