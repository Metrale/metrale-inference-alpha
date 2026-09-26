// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A one-connection loopback HTTP server for the Chat tests
//! (`chat_stream_tests`, `chat_more_tests`, `chat_history_tests`).
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, channel};

/// 2026-09-26: A running fake, and the channel its captured request arrives on.
pub(crate) struct Fake {
    pub(crate) port: u16,
    pub(crate) request: Receiver<String>,
}

/// 2026-09-26: Bind a loopback listener on an OS-chosen port, accept one
/// connection, send the raw request on `request`, then call `reply`. The
/// socket is dropped when `reply` returns, so the client then reads EOF.
pub(crate) fn serve<F>(reply: F) -> Fake
where
    F: FnOnce(&mut TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
    let port = listener.local_addr().expect("local addr").port();
    let (tx, request) = channel();
    std::thread::spawn(move || {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        let _ = tx.send(read_request(&mut sock));
        reply(&mut sock);
    });
    Fake { port, request }
}

/// 2026-09-26: Read until the headers and the declared `Content-Length` body
/// have arrived, or a read returns 0 or an error. It does not wait for EOF:
/// the client keeps its side open while it waits for the reply.
fn read_request(sock: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        if let Some(end) = super::find_subslice(&buf, b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).into_owned();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length: "))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= end + 4 + len {
                break;
            }
        }
        match sock.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// 2026-09-26: A 200 `text/event-stream` reply: one `write_all` and flush per
/// chunk, then a 2 ms pause, so a test can split a frame across the client's
/// reads.
pub(crate) fn sse(sock: &mut TcpStream, chunks: &[&[u8]]) {
    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
    let _ = sock.flush();
    for c in chunks {
        if sock.write_all(c).is_err() {
            return;
        }
        let _ = sock.flush();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// 2026-09-26: A complete response with the given status line, content type
/// and a `Content-Length` body.
pub(crate) fn refuse(sock: &mut TcpStream, status: &str, ctype: &str, body: &str) {
    let _ = sock.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    );
    let _ = sock.flush();
}

/// 2026-09-26: An OS-chosen loopback port whose listener has been dropped.
pub(crate) fn dead_port() -> u16 {
    let l = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
    l.local_addr().expect("local addr").port()
}
