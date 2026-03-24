import hashlib
import os
import urllib.request

os.makedirs('public/letters', exist_ok=True)

for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    filename = f"Sign_language_{c}.svg".encode('utf-8')
    h = hashlib.md5(filename).hexdigest()
    url = f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[0:2]}/Sign_language_{c}.svg"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            with open(f"public/letters/{c}.svg", "wb") as f:
                f.write(data)
        print(f"Downloaded {c}")
    except Exception as e:
        print(f"Failed {c}: {e}")
