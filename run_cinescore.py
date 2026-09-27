from server import PORT, HOST
from http.server import ThreadingHTTPServer
from server import Handler

if __name__ == '__main__':
    print(f'Starting CineScore at http://{HOST}:{PORT}')
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
