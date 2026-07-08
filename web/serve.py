# Bitcrush dev server. Run this in its OWN terminal so it stays up:
#     cd "C:\Users\AaryanMehta\Downloads\GH solo\web"
#     python serve.py
# Then open  http://localhost:8123/   (Ctrl+C here to stop)
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 8123
print(f"BITCRUSH BC-06 dev server  ->  http://localhost:{PORT}/   (Ctrl+C to stop)")
ThreadingHTTPServer(("127.0.0.1", PORT), SimpleHTTPRequestHandler).serve_forever()
