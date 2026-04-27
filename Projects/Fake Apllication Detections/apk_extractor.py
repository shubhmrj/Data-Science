import os
import hashlib


def extract_features(apk_path: str, top_features: list) -> dict:
    try:
        from androguard.misc import AnalyzeAPK
    except ImportError:
        raise ImportError("androguard not installed. Run: pip install androguard==4.1.2")

    a    = None
    d    = None
    dx   = None

    # Try full analysis first
    try:
        a, d, dx = AnalyzeAPK(apk_path)
    except Exception as e1:
        print(f"Full analysis failed ({e1}), trying APK-only parse...")
        try:
            from androguard.core.apk import APK
            a = APK(apk_path)
        except Exception as e2:
            raise Exception(f"Cannot parse APK: {e2}")

    # ── Safe extractions ─────────────────────────
    def safe(fn, default):
        try:
            return fn()
        except Exception:
            return default

    permissions  = safe(lambda: set(a.get_permissions()), set())
    activities   = safe(lambda: a.get_activities(), [])
    services     = safe(lambda: a.get_services(), [])
    receivers    = safe(lambda: a.get_receivers(), [])
    package_name = safe(lambda: a.get_package(), "unknown")
    app_name     = safe(lambda: a.get_app_name(), "Unknown")
    min_sdk      = safe(lambda: a.get_min_sdk_version(), "?")
    target_sdk   = safe(lambda: a.get_target_sdk_version(), "?")

    # Short permission names for matching
    short_perms = set()
    for p in permissions:
        short_perms.add(p.split('.')[-1].upper())
        short_perms.add(p.upper())

    # ── API calls ────────────────────────────────
    api_calls = set()
    if dx is not None:
        try:
            for method in dx.get_methods():
                for _, call, _ in method.get_xref_to():
                    api_calls.add(f"{call.get_class_name()}->{call.get_name()}")
        except Exception:
            pass

    # ── Feature vector ───────────────────────────
    feature_vector = {}
    for feat in top_features:
        feat_upper = feat.upper()
        matched = (
            feat_upper in short_perms
            or any(feat_upper in p for p in short_perms)
            or any(p.endswith(feat_upper) for p in short_perms)
        )
        feature_vector[feat] = 1 if matched else 0

    # ── Metadata ─────────────────────────────────
    meta = {
        "apk_name":        os.path.basename(apk_path),
        "package_name":    package_name,
        "app_name":        app_name or "Unknown",
        "min_sdk":         str(min_sdk) if min_sdk else "?",
        "target_sdk":      str(target_sdk) if target_sdk else "?",
        "permissions":     sorted(list(permissions)),
        "num_permissions": len(permissions),
        "num_activities":  len(activities),
        "num_services":    len(services),
        "num_receivers":   len(receivers),
        "file_size_kb":    round(os.path.getsize(apk_path) / 1024, 1),
        "md5":             _md5(apk_path),
        "active_features": sum(feature_vector.values()),
        "total_features":  len(feature_vector),
        "risk_signals":    _get_risk_signals(short_perms, api_calls),
    }

    return feature_vector, meta


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _get_risk_signals(permissions: set, api_calls: set) -> list:
    HIGH_RISK_PERMS = [
        'READ_SMS', 'SEND_SMS', 'RECEIVE_SMS',
        'SYSTEM_ALERT_WINDOW', 'BIND_ACCESSIBILITY_SERVICE',
        'READ_CALL_LOG', 'WRITE_CALL_LOG', 'PROCESS_OUTGOING_CALLS',
        'RECORD_AUDIO', 'READ_CONTACTS', 'WRITE_CONTACTS',
        'GET_ACCOUNTS', 'USE_CREDENTIALS', 'BIND_DEVICE_ADMIN',
        'INSTALL_PACKAGES', 'RECEIVE_BOOT_COMPLETED',
    ]
    HIGH_RISK_APIS = [
        'getDeviceId', 'getSubscriberId', 'sendTextMessage',
        'execCommand', 'DexClassLoader', 'loadClass',
    ]
    signals = []
    for p in HIGH_RISK_PERMS:
        if p in permissions:
            signals.append({'type': 'permission', 'name': p, 'severity': 'high'})
    for api in HIGH_RISK_APIS:
        if any(api.lower() in call.lower() for call in api_calls):
            signals.append({'type': 'api_call', 'name': api, 'severity': 'medium'})
    return signals[:10]