from http.server import BaseHTTPRequestHandler
import os
import mimetypes

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Extract relative clean path
        clean_path = self.path.split('?')[0].strip('/')
        if not clean_path or clean_path == '':
            clean_path = 'index.html'

        # Serve static file if found
        if os.path.exists(clean_path) and os.path.isfile(clean_path):
            ctype, _ = mimetypes.guess_type(clean_path)
            if not ctype:
                ctype = 'text/html' if clean_path.endswith('.html') else 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-type', ctype)
            self.end_headers()
            with open(clean_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            # Fallback to index.html
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            if os.path.exists('index.html'):
                with open('index.html', 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b'<h1>Retail Crash-Test Engine</h1>')
        return

app = handler
application = handler
