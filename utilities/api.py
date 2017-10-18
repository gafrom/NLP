from http.server import BaseHTTPRequestHandler
import json

msg = """
Welcome to Predictor server
===========================\n
Please, use our API as follows:\n
request: POST JSON { text: \"Text of the article\" } to `/predict`
respond: in plain text containing a predicted article's class
"""

class API(BaseHTTPRequestHandler):
  def _set_headers(self):
    self.send_response(200)
    self.send_header('Content-type', 'text/plain')
    self.end_headers()

  def do_GET(self):
    self._set_headers();
    self.wfile.write(msg.encode('utf-8'))

  def do_POST(self):
    if self.path[1:] == 'predict':
      self._set_headers();
      content_length = int(self.headers['Content-Length'])
      raw_data = self.rfile.read(content_length)
      data = json.loads(raw_data)
      print(data.__class__)
      self.wfile.write(data['text'].encode('utf-8'))
    else:
      self.send_error(404)

  def do_HEAD(self):
    self._set_headers()
