"""
MP Police Batch APK Scanner
Simple command-line tool for batch scanning APK directories
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime

class MPPoliceBatchScanner:
    """Batch scanner for MP Police APK detection"""
    
    def __init__(self, api_url="http://127.0.0.1:5000"):
        self.api_url = api_url
        self.results = []
        
    def scan_directory(self, directory_path, output_file=None):
        """Scan all APKs in a directory"""
        
        directory = Path(directory_path)
        if not directory.exists():
            print(f"[ERROR] Directory not found: {directory_path}")
            return False
        
        # Find all APK files
        apk_files = list(directory.glob("**/*.apk"))
        
        if len(apk_files) == 0:
            return True
        
        # Scan each APK
        suspicious_count = 0
        legitimate_count = 0
        error_count = 0
        
        for i, apk_file in enumerate(apk_files, 1):
            try:
                # Prepare file for upload
                with open(apk_file, 'rb') as f:
                    files = {'file': (apk_file.name, f, 'application/vnd.android.package-archive')}
                    
                    # Send to API
                    response = requests.post(f"{self.api_url}/api/analyze", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        analysis = result.get('analysis_result', {})
                        
                        classification = analysis.get('classification', 'UNKNOWN')
                        confidence = analysis.get('confidence', 0.0)
                        
                        # Count results
                        if classification == 'SUSPICIOUS':
                            suspicious_count += 1
                            status_icon = "[SUSPICIOUS]"
                        elif classification == 'LEGITIMATE':
                            legitimate_count += 1
                            status_icon = "[LEGITIMATE]"
                        else:
                            error_count += 1
                            status_icon = "[UNKNOWN]"
                        
                        # result stored (silent mode)
                        
                        # Store result
                        scan_result = {
                            'filename': apk_file.name,
                            'path': str(apk_file),
                            'size_mb': apk_file.stat().st_size / (1024 * 1024),
                            'classification': classification,
                            'confidence': confidence,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'success'
                        }
                        
                        if 'anomaly_score' in analysis:
                            scan_result['anomaly_score'] = analysis['anomaly_score']
                        
                        self.results.append(scan_result)
                        
                    else:
                        error_count += 1
                        self.results.append({
                            'filename': apk_file.name,
                            'path': str(apk_file),
                            'classification': 'ERROR',
                            'error': f"API Error: {response.status_code}",
                            'timestamp': datetime.now().isoformat(),
                            'status': 'error'
                        })
                        
            except Exception as e:
                    error_count += 1
                    self.results.append({
                        'filename': apk_file.name,
                        'path': str(apk_file),
                        'classification': 'ERROR',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error'
                    })
        # # Print summary
        # print("\n" + "=" * 50)
        # print("SCAN SUMMARY")
        # print("=" * 50)
        # print(f"Total APKs scanned: {len(apk_files)}")
        # print(f"[OK] Legitimate: {legitimate_count}")
        # print(f"[WARNING] Suspicious: {suspicious_count}")
        # print(f"[ERROR] Errors: {error_count}")
        
        if suspicious_count > 0:
            for result in self.results:
                if result.get('classification') == 'SUSPICIOUS':
                    # keep results stored; no console output
                    pass
        
        # Save results to file
        if output_file:
                self.save_results(output_file)
        return True
    
    def save_results(self, output_file):
        """Save scan results to JSON file"""
        try:
            report = {
                'scan_timestamp': datetime.now().isoformat(),
                'total_scanned': len(self.results),
                'summary': {
                    'legitimate': len([r for r in self.results if r.get('classification') == 'LEGITIMATE']),
                    'suspicious': len([r for r in self.results if r.get('classification') == 'SUSPICIOUS']),
                    'errors': len([r for r in self.results if r.get('classification') == 'ERROR'])
                },
                'results': self.results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            
        except Exception as e:
            print(f"[ERROR] Failed to save results: {str(e)}")
    
    def scan_banking_apks(self):
        """Quick scan of the banking APKs dataset"""
        banking_dir = Path(__file__).parent / "mp_police_datasets" / "legitimate" / "banking"
        output_file = Path(__file__).parent / "mp_police_scan_results.json"
        
        return self.scan_directory(str(banking_dir), str(output_file))

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("MP Police Batch APK Scanner")
        print("Usage:")
        print("  python mp_police_batch_scanner.py <directory_path> [output_file.json]")
        print("  python mp_police_batch_scanner.py --banking  # Scan banking APKs dataset")
        return
    
    scanner = MPPoliceBatchScanner()
    
    if sys.argv[1] == "--banking":
        # Quick banking APKs scan
        scanner.scan_banking_apks()
    else:
        # Custom directory scan
        directory_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        scanner.scan_directory(directory_path, output_file)

if __name__ == "__main__":
    main()
