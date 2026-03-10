import socket
import struct
import threading
from typing import Optional


def recv_exact(conn: socket.socket, n: int, stop_event: threading.Event) -> bytes:
    """Receive exactly n bytes (helper for length-prefixed TCP packets)."""
    buf = bytearray()
    while len(buf) < n:
        if stop_event.is_set():
            raise InterruptedError("Stopped by user.")
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            raise ConnectionError("Socket closed while receiving.")
        buf.extend(chunk)
    return bytes(buf)


def recv_packet(conn: socket.socket, stop_event: threading.Event) -> bytes:
    """Receive a length-prefixed packet: 4-byte big-endian length + payload."""
    header = recv_exact(conn, 4, stop_event)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid packet length: {length}")
    return recv_exact(conn, length, stop_event)


def send_packet(conn: socket.socket, payload: bytes, lock: threading.Lock, stop_event: threading.Event) -> None:
    """Send a length-prefixed packet in a thread-safe way."""
    if stop_event.is_set():
        return
    header = struct.pack(">I", len(payload))
    with lock:
        conn.sendall(header + payload)


def safe_close_conn(conn: Optional[socket.socket]) -> None:
    """Best-effort shutdown/close."""
    if conn is None:
        return
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass
