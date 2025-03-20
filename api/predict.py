from http.server import BaseHTTPRequestHandler
from src.api.app import app
from fastapi.responses import JSONResponse
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse the JSON data
            data = json.loads(post_data.decode('utf-8'))
            
            # Call the FastAPI endpoint
            response = app.predict(data)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            # Handle errors
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {"error": str(e)}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
            
    def do_GET(self):
        # Handle GET request (e.g., for health check)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = {"status": "API is running"}
        self.wfile.write(json.dumps(response).encode('utf-8')) 