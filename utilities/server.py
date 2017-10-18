from http.server import HTTPServer
from utilities.api import API

class Server(object):
  def __init__(self, port=3000):
    self.port = port

  def launch(self):
    server_address = ('', self.port)
    httpd = HTTPServer(server_address, API)
    httpd.serve_forever()
