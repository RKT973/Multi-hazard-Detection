from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from collections import deque
from pathlib import Path
import base64

app = Flask(__name__)
CORS(app)

# Keep captured frames outside the frontend folder so live-reload tools do not
# refresh the dashboard every time a new image is saved.
CAMERA_FRAME_DIR = Path(__file__).resolve().parent.parent / 'IDP_camera_frames'
CAMERA_FRAME_DIR.mkdir(exist_ok=True)
BASE_DIR = Path(__file__).resolve().parent

CV_MODEL_CONFIGS = (
    ('Fire CV model', BASE_DIR / 'fire_runs' / 'fire_classifier' / 'best_model.pt'),
    ('Smoke CV model', BASE_DIR / 'smoke_runs' / 'smoke_classifier' / 'best_model.pt'),
    ('Water leak CV model', BASE_DIR / 'water_runs' / 'leak_classifier' / 'best_model.pt'),
)

# ── BACKEND CHANGE: added 'pressure' to latest_data ──────────────────────────
latest_data = {
    'temperature': 0,
    'humidity': 0,
    'gas': 0,
    'light': 0,
    'pressure': 1013.25,   # NEW: BMP280 pressure in hPa (sea-level default)
    'hazard': 'SAFE',
    'active_hazards': [],
    'newly_detected_hazards': [],
    'cleared_hazards': [],
    'hazard_event_id': 0,
    'hazard_event_timestamp': None,
    'pending_clear_hazards': {},
    'sensor_hazard': 'SAFE',
    'cv_hazard': 'SAFE',
    'fusion_scores': {},
    'sensor_scores': {},
    'cv_results': {},
    'timestamp': None
}

# ── BACKEND CHANGE: added 'pressure' deque to history ────────────────────────
history = {
    'temperature': deque(maxlen=20),
    'humidity': deque(maxlen=20),
    'gas': deque(maxlen=20),
    'light': deque(maxlen=20),
    'pressure': deque(maxlen=20),   # NEW: pressure history
    'timestamps': deque(maxlen=20)
}

latest_sensor_state = {
    'hazard': 'SAFE',
    'scores': {},
    'timestamp': None
}

latest_cv_state = {
    'hazard': 'SAFE',
    'scores': {},
    'results': {},
    'timestamp': None
}

hazard_history = []
hazard_history_next_id = 1

SENSOR_WEIGHT = 0.7
CV_WEIGHT = 0.3
CV_MAX_AGE_SECONDS = 60
HAZARD_CLEAR_CONFIRMATION_CYCLES = 3

HAZARD_PRIORITY = {
    'SAFE': 0,
    'WARNING': 1,
    'WATER_LEAK': 2,
    'WATER_LEAK_RISK': 2,
    'SMOKE': 3,
    'SMOKE_RISK': 3,
    'GAS_LEAK': 4,
    'OVERHEATING': 4,
    'FIRE_RISK': 5,
    'DANGER': 6,
}

DISPLAY_HAZARD = {
    'DANGER': 'DANGER',
    'SMOKE': 'SMOKE_RISK',
    'WATER_LEAK': 'WATER_LEAK_RISK',
    'GAS_LEAK': 'GAS_LEAK',
    'OVERHEATING': 'OVERHEATING',
}

HAZARD_STATUS = {
    'DANGER': 'danger',
    'FIRE_RISK': 'danger',
    'GAS_LEAK': 'danger',
    'OVERHEATING': 'danger',
    'SMOKE_RISK': 'warning',
    'WATER_LEAK_RISK': 'warning',
    'WARNING': 'warning',
}


def _record_hazard_detected(hazard):
    global hazard_history_next_id
    entry = {
        'id': hazard_history_next_id,
        'hazard': hazard,
        'status': HAZARD_STATUS.get(hazard, 'warning'),
        'resolved': False,
        'detected_at': datetime.now().isoformat(),
        'resolved_at': None,
    }
    hazard_history_next_id += 1
    hazard_history.insert(0, entry)
    del hazard_history[200:]


def _record_hazard_cleared(hazard):
    for entry in hazard_history:
        if entry['hazard'] == hazard and not entry['resolved']:
            entry['resolved'] = True
            entry['resolved_at'] = datetime.now().isoformat()
            return


class HazardEventManager:
    def __init__(self, clear_confirmation_cycles=HAZARD_CLEAR_CONFIRMATION_CYCLES):
        self.clear_confirmation_cycles = clear_confirmation_cycles
        self.confirmed_active = set()
        self.pending_clear_counts = {}
        self.announced_hazards = set()
        self.event_id = 0
        self.event_timestamp = None

    def update(self, detected_hazards):
        detected = set(detected_hazards)
        newly_detected = []
        cleared = []

        for hazard in detected:
            if hazard in self.pending_clear_counts:
                del self.pending_clear_counts[hazard]
            elif hazard not in self.announced_hazards:
                newly_detected.append(hazard)
                self.confirmed_active.add(hazard)
                self.announced_hazards.add(hazard)
            elif hazard not in self.confirmed_active:
                self.confirmed_active.add(hazard)

        for hazard in list(self.confirmed_active):
            if hazard in detected:
                continue

            self.pending_clear_counts[hazard] = self.pending_clear_counts.get(hazard, 0) + 1
            if self.pending_clear_counts[hazard] >= self.clear_confirmation_cycles:
                self.confirmed_active.remove(hazard)
                self.announced_hazards.discard(hazard)
                del self.pending_clear_counts[hazard]
                cleared.append(hazard)

        active_hazards = sorted(
            self.confirmed_active,
            key=lambda item: HAZARD_PRIORITY.get(item, 0),
            reverse=True
        )
        newly_detected = sorted(
            newly_detected,
            key=lambda item: HAZARD_PRIORITY.get(item, 0),
            reverse=True
        )
        cleared = sorted(
            cleared,
            key=lambda item: HAZARD_PRIORITY.get(item, 0),
            reverse=True
        )

        if newly_detected or cleared:
            self.event_id += 1
            self.event_timestamp = datetime.now().isoformat()
            for hazard in newly_detected:
                _record_hazard_detected(hazard)
            for hazard in cleared:
                _record_hazard_cleared(hazard)

        return {
            'active_hazards': active_hazards,
            'newly_detected_hazards': newly_detected,
            'cleared_hazards': cleared,
            'hazard_event_id': self.event_id,
            'hazard_event_timestamp': self.event_timestamp,
            'pending_clear_hazards': dict(self.pending_clear_counts),
        }


class CVModelRunner:
    def __init__(self, model_configs):
        self.models = []
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except ImportError as exc:
            self.load_error = exc
            print(f"[CV] Models disabled: missing dependency: {exc}")
            return

        self.load_error = None
        self.torch = torch
        self.transforms = transforms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[CV] Using device: {self.device}")

        for label, model_path in model_configs:
            if not model_path.exists():
                print(f"[CV] {label}: model file not found at {model_path}")
                continue

            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                class_to_idx = checkpoint['class_to_idx']
                idx_to_class = {idx: name for name, idx in class_to_idx.items()}
                img_size = int(checkpoint.get('img_size', 224))

                model = models.efficientnet_v2_s(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, len(class_to_idx))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()

                preprocess = transforms.Compose([
                    transforms.Resize(int(img_size * 1.15)),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])

                self.models.append({
                    'label': label,
                    'model': model,
                    'idx_to_class': idx_to_class,
                    'preprocess': preprocess,
                    'img_size': img_size,
                    'epoch': checkpoint.get('epoch'),
                    'val_f1': checkpoint.get('val_f1'),
                })
                print(f"[CV] {label}: loaded {model_path} (img_size={img_size})")
            except Exception as exc:
                print(f"[CV] {label}: failed to load {model_path}: {exc}")

    def analyze_frame(self, frame_path):
        print(f"\n[CV] Analysing captured frame: {frame_path}")
        predictions = {}

        if self.load_error:
            print(f"[CV] Skipped all models because CV dependencies are unavailable: {self.load_error}")
            return predictions

        if not self.models:
            print("[CV] No CV models are loaded.")
            return predictions

        for model_info in self.models:
            try:
                prediction = self._predict(model_info, frame_path)
                predictions[model_info['label']] = prediction
                print(
                    f"[CV] {model_info['label']}: "
                    f"{prediction['class_name']} ({prediction['confidence']:.4f})"
                )
            except Exception as exc:
                print(f"[CV] {model_info['label']}: prediction failed: {exc}")
        return predictions

    def _predict(self, model_info, frame_path):
        from PIL import Image

        with Image.open(frame_path) as image:
            image = image.convert('RGB')
            tensor = model_info['preprocess'](image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = model_info['model'](tensor)
            probabilities = self.torch.softmax(logits, dim=1)[0]
            confidence, class_index = self.torch.max(probabilities, dim=0)

        class_index = int(class_index.item())
        return {
            'class_index': class_index,
            'class_name': model_info['idx_to_class'].get(class_index, str(class_index)),
            'confidence': float(confidence.item()),
        }


cv_model_runner = CVModelRunner(CV_MODEL_CONFIGS)
hazard_event_manager = HazardEventManager()


def _clamp_score(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _sensor_scores_from_payload(data, hazard):
    scores = {
        'DANGER': _clamp_score(data.get('fire_score')),
        'SMOKE': _clamp_score(data.get('smoke_score')),
        'GAS_LEAK': _clamp_score(data.get('gas_score')),
        'WATER_LEAK': _clamp_score(data.get('water_score')),
        'OVERHEATING': _clamp_score(data.get('heat_score')),
    }

    if not any(scores.values()) and hazard != 'SAFE':
        fallback_score = 1.0 if hazard == 'DANGER' else 0.75
        scores[hazard] = max(scores.get(hazard, 0.0), fallback_score)

    if hazard == 'WARNING':
        scores['DANGER'] = max(scores['DANGER'], 0.42)

    return scores


def _cv_scores_from_results(results):
    scores = {
        'DANGER': 0.0,
        'SMOKE': 0.0,
        'WATER_LEAK': 0.0,
    }

    for label, prediction in results.items():
        class_name = str(prediction.get('class_name', '')).lower()
        confidence = _clamp_score(prediction.get('confidence'))

        if label == 'Fire CV model':
            if class_name == 'uncontrolled_fire':
                scores['DANGER'] = max(scores['DANGER'], confidence)
            elif class_name == 'controlled_fire':
                scores['DANGER'] = max(scores['DANGER'], confidence * 0.55)
        elif label == 'Smoke CV model' and class_name == 'smoke':
            scores['SMOKE'] = max(scores['SMOKE'], confidence)
        elif label == 'Water leak CV model' and class_name == 'leak':
            scores['WATER_LEAK'] = max(scores['WATER_LEAK'], confidence * 0.65)

    return scores


def _cv_hazard_from_scores(cv_scores):
    if cv_scores.get('DANGER', 0.0) >= 0.92:
        return 'DANGER'
    if cv_scores.get('SMOKE', 0.0) >= 0.80:
        return 'SMOKE_RISK'
    if cv_scores.get('WATER_LEAK', 0.0) >= 0.55:
        return 'WATER_LEAK_RISK'
    return 'SAFE'


def _seconds_since(timestamp):
    if not timestamp:
        return None
    return max(0.0, (datetime.now() - timestamp).total_seconds())


def _cv_age_weight():
    age = _seconds_since(latest_cv_state.get('timestamp'))
    if age is None or age > CV_MAX_AGE_SECONDS:
        return 0.0
    return max(0.0, 1.0 - (age / CV_MAX_AGE_SECONDS))


def _select_hazard(fusion_scores, sensor_scores, cv_scores):
    candidates = {hazard: score for hazard, score in fusion_scores.items() if hazard != 'SAFE'}
    if not candidates:
        return 'SAFE'

    hazard, score = max(
        candidates.items(),
        key=lambda item: (item[1], HAZARD_PRIORITY.get(item[0], 0))
    )

    sensor_support = sensor_scores.get(hazard, 0.0)
    cv_support = cv_scores.get(hazard, 0.0)

    if hazard == 'WATER_LEAK':
        if sensor_support < 0.45 or score < 0.55:
            return 'SAFE'
    elif hazard == 'DANGER':
        if sensor_support < 0.40 and cv_support < 0.92:
            return 'SAFE'
        if score < 0.58:
            return 'WARNING'
    elif hazard == 'SMOKE':
        if sensor_support < 0.35 and cv_support < 0.80:
            return 'SAFE'
        if score < 0.52:
            return 'WARNING'
    elif score < 0.55:
        return 'SAFE'

    return hazard


def _hazard_is_active(hazard, score, sensor_scores, cv_scores):
    sensor_support = sensor_scores.get(hazard, 0.0)
    cv_support = cv_scores.get(hazard, 0.0)

    if hazard == 'WATER_LEAK':
        return sensor_support >= 0.45 and score >= 0.55
    if hazard == 'DANGER':
        return score >= 0.58 and (sensor_support >= 0.40 or cv_support >= 0.92)
    if hazard == 'SMOKE':
        return score >= 0.52 and (sensor_support >= 0.35 or cv_support >= 0.80)
    return score >= 0.55


def _active_hazards_from_scores(fusion_scores, sensor_scores, cv_scores):
    active = []
    for hazard, score in fusion_scores.items():
        if _hazard_is_active(hazard, score, sensor_scores, cv_scores):
            active.append(DISPLAY_HAZARD.get(hazard, hazard))

    return sorted(active, key=lambda item: HAZARD_PRIORITY.get(item, 0), reverse=True)


def update_fused_hazard():
    sensor_scores = latest_sensor_state.get('scores', {})
    cv_scores = latest_cv_state.get('scores', {})
    cv_weight = CV_WEIGHT * _cv_age_weight()
    sensor_weight = SENSOR_WEIGHT + (CV_WEIGHT - cv_weight)

    hazards = {'DANGER', 'SMOKE', 'GAS_LEAK', 'WATER_LEAK', 'OVERHEATING'}
    fusion_scores = {}
    for hazard in hazards:
        fusion_scores[hazard] = round(
            sensor_weight * sensor_scores.get(hazard, 0.0)
            + cv_weight * cv_scores.get(hazard, 0.0),
            4
        )

    detected_hazards = _active_hazards_from_scores(fusion_scores, sensor_scores, cv_scores)
    hazard_events = hazard_event_manager.update(detected_hazards)
    active_hazards = hazard_events['active_hazards']
    display_hazard = active_hazards[0] if active_hazards else 'SAFE'

    latest_data.update({
        'hazard': display_hazard,
        'active_hazards': active_hazards,
        'newly_detected_hazards': hazard_events['newly_detected_hazards'],
        'cleared_hazards': hazard_events['cleared_hazards'],
        'hazard_event_id': hazard_events['hazard_event_id'],
        'hazard_event_timestamp': hazard_events['hazard_event_timestamp'],
        'pending_clear_hazards': hazard_events['pending_clear_hazards'],
        'sensor_hazard': latest_sensor_state.get('hazard', 'SAFE'),
        'cv_hazard': latest_cv_state.get('hazard', 'SAFE'),
        'fusion_scores': fusion_scores,
        'sensor_scores': sensor_scores,
        'cv_results': latest_cv_state.get('results', {}),
    })

    print(
        "[FUSION] sensor="
        f"{latest_sensor_state.get('hazard', 'SAFE')} {sensor_scores} | "
        f"cv={latest_cv_state.get('hazard', 'SAFE')} {cv_scores} | "
        f"detected={detected_hazards or ['SAFE']} "
        f"active={active_hazards or ['SAFE']} "
        f"new={hazard_events['newly_detected_hazards']} "
        f"cleared={hazard_events['cleared_hazards']} "
        f"primary={display_hazard} {fusion_scores}"
    )


@app.route('/data', methods=['POST'])
def receive_data():
    try:
        print("\n--- Incoming Request ---")
        print("RAW:", request.data)

        data = request.get_json(force=True, silent=True)

        print("PARSED JSON:", data)

        if data is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid JSON format'
            }), 400

        # Safe extraction
        temp     = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        gas      = float(data.get('gas', 0))
        light    = float(data.get('light', 0))
        hazard   = str(data.get('hazard', 'SAFE'))

        # ── BACKEND CHANGE: safely parse pressure from ESP32 JSON ─────────────
        pressure = float(data.get('pressure', 1013.25))   # NEW
        timestamp = datetime.now().isoformat()
        latest_sensor_state.update({
            'hazard': hazard,
            'scores': _sensor_scores_from_payload(data, hazard),
            'timestamp': datetime.now()
        })
        latest_data.update({
            'temperature': temp,
            'humidity':    humidity,
            'gas':         gas,
            'light':       light,
            'pressure':    pressure,   # NEW
            'sensor_hazard': hazard,
            'timestamp':   timestamp
        })

        history['temperature'].append(temp)
        history['humidity'].append(humidity)
        history['gas'].append(gas)
        history['light'].append(light)
        history['pressure'].append(pressure)   # NEW
        history['timestamps'].append(latest_data['timestamp'])
        update_fused_hazard()

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/latest', methods=['GET'])
def get_latest():
    # ── BACKEND CHANGE: pressure included in /latest response ────────────────
    return jsonify({
        'current': latest_data,
        'history': {
            'temperature': list(history['temperature']),
            'humidity':    list(history['humidity']),
            'gas':         list(history['gas']),
            'light':       list(history['light']),
            'pressure':    list(history['pressure']),   # NEW
            'timestamps':  list(history['timestamps'])
        }
    }), 200


@app.route('/hazard-history', methods=['GET'])
def get_hazard_history():
    return jsonify({
        'history': hazard_history,
        'active_hazards': latest_data.get('active_hazards', []),
        'current_hazard': latest_data.get('hazard', 'SAFE')
    }), 200


@app.route('/camera-frame', methods=['POST'])
def receive_camera_frame():
    try:
        data = request.get_json(force=True, silent=True)

        if not data or 'image' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing image data'
            }), 400

        if request.headers.get('X-IDP-Camera-Source') != 'website-camera' or data.get('source') != 'website-camera':
            print(f"[CV] Rejected non-website camera upload from {request.remote_addr}")
            return jsonify({
                'status': 'ignored',
                'message': 'Only website camera uploads are accepted'
            }), 403

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        frame_bytes = base64.b64decode(image_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'camera_frame_{timestamp}.jpg'
        frame_path = CAMERA_FRAME_DIR / filename
        frame_path.write_bytes(frame_bytes)
        print(f"[CV] Accepted website camera frame from {request.remote_addr} ({data.get('device', 'unknown')})")
        cv_results = cv_model_runner.analyze_frame(frame_path)
        cv_scores = _cv_scores_from_results(cv_results)
        latest_cv_state.update({
            'hazard': _cv_hazard_from_scores(cv_scores),
            'scores': cv_scores,
            'results': cv_results,
            'timestamp': datetime.now()
        })
        update_fused_hazard()

        return jsonify({
            'status': 'success',
            'filename': filename,
            'path': str(frame_path),
            'cv_results': cv_results,
            'cv_scores': cv_scores,
            'fused_hazard': latest_data['hazard'],
            'active_hazards': latest_data['active_hazards'],
            'newly_detected_hazards': latest_data['newly_detected_hazards'],
            'cleared_hazards': latest_data['cleared_hazards'],
            'hazard_event_id': latest_data['hazard_event_id'],
            'hazard_event_timestamp': latest_data['hazard_event_timestamp'],
            'pending_clear_hazards': latest_data['pending_clear_hazards']
        }), 200

    except Exception as e:
        print("CAMERA ERROR:", str(e))
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/')
def home():
    return jsonify({'status': 'running'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
