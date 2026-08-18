import urllib.request
import os

urls = {
    'static/leaflet.css': 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css',
    'static/leaflet.js': 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js',
    'static/leaflet-heat.js': 'https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js',
    'static/chart.umd.js': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.js',
    'static/lucide.min.js': 'https://unpkg.com/lucide@0.408.0/dist/umd/lucide.min.js'
}

def download_web_assets(force=False):
    missing_assets = []
    for path in urls:
        if not os.path.exists(path):
            missing_assets.append(path)
            
    if not missing_assets and not force:
        print("All required web assets are already present in static/.")
        return True
        
    print(f"=== Downloading Web Assets locally to static/ (force={force}) ===")
    os.makedirs('static', exist_ok=True)
    
    success = True
    failed_downloads = []
    for path, url in urls.items():
        if force or not os.path.exists(path):
            print(f"Downloading: {url} -> {path}")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                # Use a timeout so we do not hang indefinitely
                with urllib.request.urlopen(req, timeout=15) as response, open(path, 'wb') as out_file:
                    out_file.write(response.read())
                print("  [SUCCESS]")
            except Exception as e:
                print(f"  [FAILED] {e}")
                failed_downloads.append((url, str(e)))
                success = False
                
    if not success:
        error_msg = f"Failed to download some assets: {failed_downloads}"
        print(f"CRITICAL ERROR: {error_msg}")
        raise RuntimeError(error_msg)
        
    print("Done downloading assets.")
    return True

if __name__ == '__main__':
    download_web_assets(force=True)

