"""Local-only RPC using IMPRINT's existing NumPy JSON transport."""
import socket

from serve_robomme_tcp import json_to_numpy, numpy_to_json, recv_exact


class Client:
    def __init__(self, port):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=1800)

    def call(self, command, payload=None):
        raw = numpy_to_json({'cmd': command, 'obs': payload}).encode()
        self.socket.sendall(len(raw).to_bytes(4, 'big') + raw)
        size = int.from_bytes(recv_exact(self.socket, 4), 'big')
        response = json_to_numpy(recv_exact(self.socket, size).decode())
        if 'error' in response:
            raise RuntimeError(response.get('traceback', response['error']))
        return response['res']

    def close(self):
        self.socket.close()
