"""
Production-Ready Banking APK Detection API
Compatible with newly trained 18-feature banking anomaly model
Integrated with real APK analysis using androguard
"""

import os
import sys
import json
import logging
import joblib
import sqlite3
import hashlib
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def startup_checks(preferred_sklearn_version: str = "1.3.0") -> None:
    """Light, non-verbose runtime checks. Stores compatibility notes for operators.

    This function does not print to stdout to keep production output clean.
    """
    messages = []
    try:
        import sklearn
        installed = getattr(sklearn, "__version__", "unknown")
        if installed != preferred_sklearn_version:
            messages.append(f"scikit-learn version mismatch: installed={installed}, recommended={preferred_sklearn_version}")
            messages.append("Conda (recommended): conda install -c conda-forge scikit-learn=%s" % preferred_sklearn_version)
            messages.append("Or create env: conda create -n skenv python=3.10 -y && conda activate skenv && conda install -c conda-forge scikit-learn=%s joblib numpy -y" % preferred_sklearn_version)
            messages.append("Pip (if wheel available): pip install 'scikit-learn==%s' --only-binary=:all:" % preferred_sklearn_version)
    except ImportError:
        messages.append("scikit-learn not installed. Install scikit-learn to run the ML model.")

    # Save notes to a module-level var for operators / logging
    global COMPATIBILITY_NOTE
    COMPATIBILITY_NOTE = "\n".join(messages)

# configure a module logger (no default stdout noise)
logger = logging.getLogger("detecting_fake_banking")


# APK Static Analysis Module (merged here for single-file deployment)
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import zipfile
import xml.etree.ElementTree as ET

try:
    from androguard.core.bytecodes.apk import APK
    from androguard.core.bytecodes.dvm import DalvikVMFormat
    from androguard.core.analysis.analysis import Analysis
except ImportError:
    APK = None


@dataclass
class APKAnalysisResult:
    """Container for APK analysis results"""
    package_name: str
    app_name: str
    version_name: str
    version_code: int
    permissions: List[str]
    activities: List[str]
    services: List[str]
    receivers: List[str]
    certificates: List[Dict[str, Any]]
    file_hashes: Dict[str, str]
    suspicious_permissions: List[str]
    network_security_config: Optional[Dict[str, Any]]
    features: Dict[str, Any]
    risk_score: float


class APKAnalyzer:
    """Main APK analysis class (merged from analysis/apk_analyzer.py)"""
    # Suspicious permissions commonly found in banking malware
    SUSPICIOUS_PERMISSIONS = {
        'android.permission.SEND_SMS': 'high',
        'android.permission.READ_SMS': 'high',
        'android.permission.RECEIVE_SMS': 'high',
        'android.permission.CALL_PHONE': 'medium',
        'android.permission.READ_PHONE_STATE': 'medium',
        'android.permission.SYSTEM_ALERT_WINDOW': 'high',
        'android.permission.WRITE_EXTERNAL_STORAGE': 'low',
        'android.permission.ACCESS_FINE_LOCATION': 'medium',
        'android.permission.CAMERA': 'medium',
        'android.permission.RECORD_AUDIO': 'medium',
        'android.permission.GET_ACCOUNTS': 'high',
        'android.permission.AUTHENTICATE_ACCOUNTS': 'high',
        'android.permission.DEVICE_ADMIN': 'high',
        'android.permission.BIND_DEVICE_ADMIN': 'high'
    }

    # Legitimate banking app indicators
    BANKING_KEYWORDS = [
        'bank', 'banking', 'finance', 'payment', 'wallet', 'money',
        'credit', 'debit', 'account', 'transaction', 'transfer'
    ]

    def __init__(self):
        self.apk = None

    def analyze(self, apk_path: str) -> APKAnalysisResult:
        """
        Perform comprehensive APK analysis
        """
        try:
            if APK is None:
                raise Exception('androguard not available')

            self.apk = APK(apk_path)

            # Extract basic information
            package_name = self.apk.get_package()
            app_name = self.apk.get_app_name()
            version_name = self.apk.get_androidversion_name()
            version_code = self.apk.get_androidversion_code()

            # Extract permissions
            permissions = self.apk.get_permissions() or []

            # Extract components
            activities = self.apk.get_activities() or []
            services = self.apk.get_services() or []
            receivers = self.apk.get_receivers() or []

            # Analyze certificates
            certificates = self._analyze_certificates()

            # Calculate file hashes
            file_hashes = self._calculate_hashes(apk_path)

            # Identify suspicious permissions
            suspicious_permissions = self._identify_suspicious_permissions(permissions)

            # Analyze network security config
            network_config = self._analyze_network_security()

            # Extract features for ML
            features = self._extract_features(
                package_name, app_name, permissions, activities,
                services, receivers, certificates
            )

            # Calculate risk score
            risk_score = self._calculate_risk_score(
                permissions, suspicious_permissions, certificates, features
            )

            return APKAnalysisResult(
                package_name=package_name,
                app_name=app_name,
                version_name=version_name,
                version_code=version_code,
                permissions=permissions,
                activities=activities,
                services=services,
                receivers=receivers,
                certificates=certificates,
                file_hashes=file_hashes,
                suspicious_permissions=suspicious_permissions,
                network_security_config=network_config,
                features=features,
                risk_score=risk_score
            )

        except Exception as e:
            raise Exception(f"APK analysis failed: {str(e)}")

    def _analyze_certificates(self) -> List[Dict[str, Any]]:
        """Analyze APK certificates and signatures"""
        certificates = []

        try:
            for cert in self.apk.get_certificates():
                cert_info = {
                    'subject': str(cert.subject),
                    'issuer': str(cert.issuer),
                    'serial_number': str(cert.serial_number),
                    'not_valid_before': cert.not_valid_before.isoformat(),
                    'not_valid_after': cert.not_valid_after.isoformat(),
                    'signature_algorithm': cert.signature_algorithm_oid._name,
                    'is_self_signed': cert.subject == cert.issuer,
                    'key_size': cert.public_key().key_size if hasattr(cert.public_key(), 'key_size') else None
                }
                certificates.append(cert_info)
        except Exception as e:
            logger.debug("Certificate analysis error: %s", e)

        return certificates

    def _calculate_hashes(self, apk_path: str) -> Dict[str, str]:
        """Calculate various hashes of the APK file"""
        hashes = {}

        try:
            with open(apk_path, 'rb') as f:
                content = f.read()
                hashes['md5'] = hashlib.md5(content).hexdigest()
                hashes['sha1'] = hashlib.sha1(content).hexdigest()
                hashes['sha256'] = hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.debug("Hash calculation error: %s", e)

        return hashes

    def _identify_suspicious_permissions(self, permissions: List[str]) -> List[str]:
        """Identify suspicious permissions"""
        suspicious = []
        for perm in permissions:
            if perm in self.SUSPICIOUS_PERMISSIONS:
                suspicious.append(perm)
        return suspicious

    def _analyze_network_security(self) -> Optional[Dict[str, Any]]:
        """Analyze network security configuration"""
        try:
            # Look for network security config file
            network_config = self.apk.get_file("res/xml/network_security_config.xml")
            if network_config:
                # Parse XML and extract security settings
                root = ET.fromstring(network_config)
                config = {
                    'clear_traffic_permitted': root.get('cleartextTrafficPermitted', 'true'),
                    'trust_anchors': [],
                    'domain_configs': []
                }
                return config
        except Exception as e:
            logger.debug("Network security analysis error: %s", e)

        return None

    def _extract_features(self, package_name: str, app_name: str, permissions: List[str],
                         activities: List[str], services: List[str], receivers: List[str],
                         certificates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features for machine learning"""

        features = {
            # Basic features
            'permission_count': len(permissions),
            'activity_count': len(activities),
            'service_count': len(services),
            'receiver_count': len(receivers),

            # Permission-based features
            'has_sms_permissions': any('SMS' in p for p in permissions),
            'has_phone_permissions': any('PHONE' in p for p in permissions),
            'has_location_permissions': any('LOCATION' in p for p in permissions),
            'has_camera_permissions': any('CAMERA' in p for p in permissions),
            'has_admin_permissions': any('ADMIN' in p for p in permissions),
            'has_system_alert': 'android.permission.SYSTEM_ALERT_WINDOW' in permissions,

            # Suspicious permission ratio
            'suspicious_permission_ratio': len(self._identify_suspicious_permissions(permissions)) / max(len(permissions), 1),

            # Certificate features
            'certificate_count': len(certificates),
            'has_self_signed_cert': any(cert.get('is_self_signed', False) for cert in certificates),
            'cert_validity_days': self._calculate_cert_validity(certificates),

            # Name-based features
            'package_name_length': len(package_name),
            'app_name_length': len(app_name),
            'has_banking_keywords': any(keyword in app_name.lower() or keyword in package_name.lower()
                                      for keyword in self.BANKING_KEYWORDS),
            'package_name_suspicious': self._is_package_name_suspicious(package_name),

            # Component ratios
            'service_to_activity_ratio': len(services) / max(len(activities), 1),
            'receiver_to_activity_ratio': len(receivers) / max(len(activities), 1),
        }

        return features

    def _calculate_cert_validity(self, certificates: List[Dict[str, Any]]) -> int:
        """Calculate average certificate validity period in days"""
        if not certificates:
            return 0

        total_days = 0
        for cert in certificates:
            try:
                from datetime import datetime
                start = datetime.fromisoformat(cert['not_valid_before'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(cert['not_valid_after'].replace('Z', '+00:00'))
                days = (end - start).days
                total_days += days
            except:
                continue

        return total_days // max(len(certificates), 1)

    def _is_package_name_suspicious(self, package_name: str) -> bool:
        """Check if package name looks suspicious"""
        suspicious_patterns = [
            'com.android.', 'android.', 'system.', 'google.', 'samsung.',
            'temp.', 'test.', 'fake.', 'malware.'
        ]

        # Check for suspicious patterns
        for pattern in suspicious_patterns:
            if package_name.startswith(pattern) and not self._is_legitimate_system_app(package_name):
                return True

        # Check for random-looking package names
        parts = package_name.split('.')
        if len(parts) < 2:
            return True

        # Check for very short or very long package names
        if len(package_name) < 10 or len(package_name) > 100:
            return True

        return False

    def _is_legitimate_system_app(self, package_name: str) -> bool:
        """Check if this is a legitimate system app"""
        legitimate_prefixes = [
            'com.android.chrome', 'com.android.vending', 'com.google.android',
            'com.samsung.android', 'com.android.settings'
        ]

        return any(package_name.startswith(prefix) for prefix in legitimate_prefixes)

    def _calculate_risk_score(self, permissions: List[str], suspicious_permissions: List[str],
                            certificates: List[Dict[str, Any]], features: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100)"""
        risk_score = 0.0

        # Permission-based risk
        if suspicious_permissions:
            risk_score += min(len(suspicious_permissions) * 10, 40)

        # Certificate-based risk
        if features.get('has_self_signed_cert', False):
            risk_score += 20

        if features.get('cert_validity_days', 0) < 30:
            risk_score += 15

        # Package name risk
        if features.get('package_name_suspicious', False):
            risk_score += 15

        # Excessive permissions
        if features.get('permission_count', 0) > 20:
            risk_score += 10

        return min(risk_score, 100.0)

app = Flask(__name__)
CORS(app)

class ProductionBankingDetector:
    """Production Banking APK Detector with 18-feature model"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.models_dir = self.base_dir / "models"
        self.db_path = self.base_dir / "mp_police_datasets" / "apk_database.db"
        
        # Initialize APK Analyzer for real feature extraction
        self.analyzer = APKAnalyzer()
        
        # Load the newly trained banking anomaly model
        self.model = None
        self.scaler = None
        self.load_banking_model()
        
        # detector initialized
    
    def load_banking_model(self):
        """Load the newly trained banking anomaly model (18 features)"""
        try:
            model_path = self.models_dir / 'banking_anomaly_model.pkl'
            scaler_path = self.models_dir / 'banking_scaler.pkl'
            metadata_path = self.models_dir / 'banking_model_metadata.json'
            
            if model_path.exists() and scaler_path.exists():
                try:
                    # Load model while suppressing non-critical warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.model = joblib.load(model_path)
                        self.scaler = joblib.load(scaler_path)
                    # metadata load is optional
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                _ = json.load(f)
                        except Exception:
                            pass
                    return True
                except Exception as load_err:
                    logger.error("Failed to unpickle model: %s", load_err)
                    logger.error("Hint: check scikit-learn version compatibility with the training environment.")
                    return False
            else:
                logger.error("Model files not found: %s or %s", model_path, scaler_path)
                return False
                
        except Exception as e:
            logger.error("Failed to load banking model: %s", e)
            return False
    
    def extract_apk_features(self, apk_path):
        """Extract 18 features from APK file using real androguard analysis"""
        try:
            # Perform real APK analysis
            analysis_result = self.analyzer.analyze(apk_path)
            features_dict = analysis_result.features
            
            # Extract 18 features in the exact order expected by the model
            features = [
                os.path.getsize(apk_path) / (1024 * 1024),                    # 1. File size in MB
                features_dict.get('permission_count', 0),                      # 2. Permission count
                int(features_dict.get('suspicious_permission_ratio', 0) * 100), # 3. Suspicious permission percentage
                features_dict.get('activity_count', 0),                        # 4. Activity count
                features_dict.get('service_count', 0),                         # 5. Service count
                features_dict.get('receiver_count', 0),                        # 6. Receiver count
                features_dict.get('certificate_count', 0),                     # 7. Certificate count
                analysis_result.risk_score,                                    # 8. Risk score (0-100)
                1 if 'android.permission.INTERNET' in analysis_result.permissions else 0,  # 9. Has internet
                1 if features_dict.get('has_sms_permissions', False) else 0,   # 10. Has SMS permissions
                1 if features_dict.get('has_location_permissions', False) else 0, # 11. Has location
                1 if features_dict.get('has_camera_permissions', False) else 0,  # 12. Has camera
                1 if 'android.permission.WRITE_EXTERNAL_STORAGE' in analysis_result.permissions else 0, # 13. Has storage
                1 if features_dict.get('has_banking_keywords', False) else 0,   # 14. Has banking keywords
                features_dict.get('suspicious_permission_ratio', 0),            # 15. Suspicious permission ratio
                features_dict.get('activity_count', 0) + features_dict.get('service_count', 0) + features_dict.get('receiver_count', 0), # 16. Total components
                features_dict.get('package_name_length', 0),                   # 17. Package name length
                1 if features_dict.get('has_banking_keywords', False) else 0    # 18. Is banking-related
            ]
            
            # real APK analysis completed
            return np.array(features).reshape(1, -1), analysis_result
            
        except Exception as e:
            logger.error("Feature extraction failed for %s: %s", apk_path, e)
            logger.debug("Falling back to basic feature extraction for %s", apk_path)
            basic = self._extract_basic_features(apk_path)
            return basic, None
    
    def _extract_basic_features(self, apk_path):
        """Fallback: Extract basic features when androguard fails"""
        try:
            file_size = os.path.getsize(apk_path) / (1024 * 1024)
            file_name = os.path.basename(apk_path).lower()
            
            # Basic feature extraction without androguard
            features = [
                file_size, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, len(file_name), 0
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error("Basic feature extraction failed for %s: %s", apk_path, e)
            return None
    
    def classify_apk(self, apk_path):
        """Classify APK using the banking anomaly model"""
        try:
            if self.model is None or self.scaler is None:
                return {
                    'error': 'Banking model not loaded',
                    'classification': 'UNKNOWN',
                    'confidence': 0.0
                }
            
            # Extract features (and analysis result when available)
            features, analysis_result = self.extract_apk_features(apk_path)
            if features is None:
                return {
                    'error': 'Feature extraction failed',
                    'classification': 'ERROR',
                    'confidence': 0.0
                }
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(scaled_features)[0]
            anomaly_score = self.model.decision_function(scaled_features)[0]
            
            # Interpret results
            is_legitimate = prediction == 1
            confidence = abs(anomaly_score)
            
            classification = 'LEGITIMATE' if is_legitimate else 'SUSPICIOUS'
            
            result = {
                'classification': classification,
                'confidence': float(confidence),
                'anomaly_score': float(anomaly_score),
                'is_legitimate': bool(is_legitimate),
                'model_version': 'banking_anomaly_v20250901',
                'features_count': int(features.shape[1])
            }

            # Add analysis metadata when available
            if analysis_result is not None:
                result.update({
                    'package_name': getattr(analysis_result, 'package_name', None),
                    'app_name': getattr(analysis_result, 'app_name', None),
                    'risk_score': float(getattr(analysis_result, 'risk_score', 0.0)),
                    'suspicious_permissions': getattr(analysis_result, 'suspicious_permissions', [])
                })

            return result
            
        except Exception as e:
            return {
                'error': f'Classification failed: {str(e)}',
                'classification': 'ERROR',
                'confidence': 0.0
            }
    
    def log_detection(self, apk_path, result):
        """Log detection result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create detection log table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    apk_path TEXT NOT NULL,
                    apk_hash TEXT,
                    classification TEXT NOT NULL,
                    confidence REAL,
                    anomaly_score REAL,
                    model_version TEXT
                )
            ''')
            
            # Calculate APK hash
            apk_hash = hashlib.sha256(open(apk_path, 'rb').read()).hexdigest()[:16]
            
            # Insert detection log
            cursor.execute('''
                INSERT INTO detection_log 
                (timestamp, apk_path, apk_hash, classification, confidence, anomaly_score, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                apk_path,
                apk_hash,
                result.get('classification', 'UNKNOWN'),
                result.get('confidence', 0.0),
                result.get('anomaly_score', 0.0),
                result.get('model_version', 'unknown')
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error("Failed to log detection to DB: %s", e)
            return False

# Initialize detector
startup_checks()
detector = ProductionBankingDetector()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': 'production_v1.0'
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_apk():
    """Analyze uploaded APK file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.apk'):
            return jsonify({'error': 'File must be an APK'}), 400
        
        # Save uploaded file temporarily with safer handling
        temp_dir = tempfile.gettempdir()
        # Use a unique temp filename to avoid conflicts
        import uuid
        safe_filename = f"apk_{uuid.uuid4().hex}_{file.filename}"
        temp_path = os.path.join(temp_dir, safe_filename)
        
        try:
            file.save(temp_path)
            
            # Classify APK
            result = detector.classify_apk(temp_path)
            
            # Log detection
            detector.log_detection(temp_path, result)
            
        finally:
            # Safe cleanup - handle file lock errors
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning("Could not delete temp file %s: %s", temp_path, cleanup_error)
                # File will be cleaned by OS eventually
        
        # Use real analysis metadata when provided by the classifier
        package_name = result.get('package_name') if isinstance(result, dict) else None
        app_name = result.get('app_name') if isinstance(result, dict) else None
        risk_score = result.get('risk_score') if isinstance(result, dict) else None

        return jsonify({
            'status': 'success',
            'filename': file.filename,
            'analysis': {
                'package_name': package_name or f"com.{file.filename.split('.')[0].lower()}",
                'app_name': app_name or file.filename.split('.')[0],
                'version_name': '1.0',
                'is_suspicious': result.get('classification') == 'SUSPICIOUS' if isinstance(result, dict) else False,
                'security_analysis': {
                    'risk_score': int(risk_score) if risk_score is not None else int(result.get('confidence', 0) * 100)
                },
                'permission_count': result.get('permission_count', None) if isinstance(result, dict) else None,
                'suspicious_permissions': result.get('suspicious_permissions', []),
                'critical_permissions': [],
                'ml_prediction': {
                    'prediction': result.get('classification', 'UNKNOWN'),
                    'confidence': result.get('confidence', 0.0)
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/batch-scan', methods=['POST'])
def batch_scan():
    """Batch scan APK directory"""
    try:
        data = request.get_json()
        directory_path = data.get('directory_path')
        
        if not directory_path or not os.path.exists(directory_path):
            return jsonify({'error': 'Invalid directory path'}), 400
        
        results = []
        apk_files = list(Path(directory_path).glob("*.apk"))
        
        for apk_file in apk_files[:10]:  # Limit to 10 files for demo
            result = detector.classify_apk(str(apk_file))
            detector.log_detection(str(apk_file), result)
            
            results.append({
                'filename': apk_file.name,
                'path': str(apk_file),
                'result': result
            })
        
        return jsonify({
            'success': True,
            'scanned_count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Batch scan failed: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detection statistics"""
    try:
        conn = sqlite3.connect(detector.db_path)
        cursor = conn.cursor()
        
        # Get detection counts
        cursor.execute('SELECT classification, COUNT(*) FROM detection_log GROUP BY classification')
        classification_counts = dict(cursor.fetchall())
        
        # Get recent detections
        cursor.execute('''
            SELECT timestamp, apk_path, classification, confidence 
            FROM detection_log 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_detections = [
            {
                'timestamp': row[0],
                'apk_path': row[1],
                'classification': row[2],
                'confidence': row[3]
            }
            for row in cursor.fetchall()
        ]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'classification_counts': classification_counts,
            'recent_detections': recent_detections,
            'model_info': (
                (lambda: json.load(open(detector.models_dir / 'banking_model_metadata.json')))
                if (detector.models_dir / 'banking_model_metadata.json').exists() else {
                    'version': 'banking_anomaly_v20250901',
                    'features': 18,
                    'type': 'IsolationForest'
                }
            )
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Stats retrieval failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (for Render/Heroku) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # start the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)
