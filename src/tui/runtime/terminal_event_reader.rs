use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, TryRecvError};
use std::thread::JoinHandle;
use std::time::Duration;

use crossterm::event::{self, Event};

const INPUT_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Collects terminal events independently so rendering cannot prevent input capture.
pub(super) struct TerminalEventReader {
    receiver: Receiver<io::Result<Event>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl TerminalEventReader {
    pub(super) fn spawn() -> Self {
        let (sender, receiver) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            while !reader_stop.load(Ordering::Acquire) {
                match event::poll(INPUT_POLL_INTERVAL) {
                    Ok(false) => {}
                    Ok(true) => {
                        let result = event::read();
                        let failed = result.is_err();
                        if sender.send(result).is_err() || failed {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        break;
                    }
                }
            }
        });

        Self {
            receiver,
            stop,
            thread: Some(thread),
        }
    }

    #[cfg(test)]
    pub(super) fn from_receiver(receiver: Receiver<io::Result<Event>>) -> Self {
        Self {
            receiver,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        }
    }

    pub(super) fn try_recv(&self) -> io::Result<Option<Event>> {
        match self.receiver.try_recv() {
            Ok(event) => event.map(Some),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "terminal event reader stopped",
            )),
        }
    }

    pub(super) fn recv_timeout(&self, timeout: Duration) -> io::Result<Option<Event>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(event) => event.map(Some),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "terminal event reader stopped",
            )),
        }
    }
}

impl Drop for TerminalEventReader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn queued_events_remain_fifo() {
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Ok(Event::Resize(80, 24)))
            .expect("queue first event");
        sender
            .send(Ok(Event::Paste("second".into())))
            .expect("queue second event");
        let reader = TerminalEventReader::from_receiver(receiver);

        assert!(matches!(reader.try_recv(), Ok(Some(Event::Resize(80, 24)))));
        assert!(matches!(reader.try_recv(), Ok(Some(Event::Paste(text))) if text == "second"));
        assert!(matches!(reader.try_recv(), Ok(None)));
    }

    #[test]
    fn reader_error_is_forwarded() {
        let (sender, receiver) = mpsc::channel();
        sender
            .send(Err(io::Error::other("input failed")))
            .expect("queue reader error");
        let reader = TerminalEventReader::from_receiver(receiver);

        let error = reader.try_recv().expect_err("reader error is surfaced");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(error.to_string(), "input failed");
    }
}
