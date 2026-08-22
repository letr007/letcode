//! Platform process-tree containment for spawned tool and MCP processes.

#[cfg(windows)]
use anyhow::Context;
use anyhow::{Result, anyhow};
use tokio::process::{Child, Command};

pub(crate) struct ManagedChild {
    child: Option<Child>,
    containment: Containment,
}

impl ManagedChild {
    pub(crate) fn spawn(command: Command) -> Result<Self> {
        Containment::spawn(command)
    }

    pub(crate) fn take_stdin(&mut self) -> Option<tokio::process::ChildStdin> {
        self.child.as_mut()?.stdin.take()
    }

    pub(crate) fn take_stdout(&mut self) -> Option<tokio::process::ChildStdout> {
        self.child.as_mut()?.stdout.take()
    }

    pub(crate) fn take_stderr(&mut self) -> Option<tokio::process::ChildStderr> {
        self.child.as_mut()?.stderr.take()
    }

    pub(crate) async fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child
            .as_mut()
            .ok_or_else(|| anyhow!("child is no longer managed"))?
            .wait()
            .await
            .map_err(Into::into)
    }

    pub(crate) fn terminate(&mut self) {
        let contained = self.containment.terminate();
        if !contained && let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }

    pub(crate) fn terminate_gracefully(&mut self) {
        self.containment.terminate_gracefully();
    }

    pub(crate) async fn terminate_and_wait(&mut self) -> Result<std::process::ExitStatus> {
        self.terminate();
        tokio::time::timeout(std::time::Duration::from_secs(5), self.wait())
            .await
            .map_err(|_| anyhow!("timed out waiting for terminated process tree"))?
    }

    pub(crate) fn disarm(&mut self) {
        if let Err(error) = self.containment.finish_normally() {
            tracing::warn!(%error, "failed to release process-tree containment after normal exit");
        }
        self.child.take();
    }

    pub(crate) fn spawn_reaper(mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = child.wait().await;
                drop(self);
            });
        }
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.terminate();
        let Some(mut child) = self.child.take() else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = child.wait().await;
            });
        }
    }
}

#[cfg(unix)]
struct Containment {
    process_group: Option<i32>,
}

#[cfg(unix)]
impl Containment {
    fn spawn(mut command: Command) -> Result<ManagedChild> {
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn()?;
        let process_group = child.id().map(|pid| pid as i32);
        Ok(ManagedChild {
            child: Some(child),
            containment: Self { process_group },
        })
    }

    fn terminate(&mut self) -> bool {
        if let Some(process_group) = self.process_group {
            unsafe {
                libc::kill(-process_group, libc::SIGKILL);
            }
            true
        } else {
            false
        }
    }

    fn terminate_gracefully(&mut self) {
        if let Some(process_group) = self.process_group {
            unsafe {
                libc::kill(-process_group, libc::SIGTERM);
            }
        }
    }

    fn finish_normally(&mut self) -> std::io::Result<()> {
        self.process_group = None;
        Ok(())
    }
}

#[cfg(windows)]
struct Containment {
    job: Option<WindowsJob>,
}

#[cfg(windows)]
impl Containment {
    fn spawn(mut command: Command) -> Result<ManagedChild> {
        use windows_sys::Win32::System::Threading::CREATE_SUSPENDED;

        let job = WindowsJob::create().context("failed to create Windows process job")?;
        command.creation_flags(CREATE_SUSPENDED);
        command.kill_on_drop(true);
        let mut child = command.spawn()?;
        let process_id = child
            .id()
            .ok_or_else(|| anyhow!("spawned Windows child has no process id"))?;
        if let Err(error) = job.assign_and_resume(process_id) {
            let process_handle = WindowsJob::open_process_handle(process_id).ok();
            if let Some(process_handle) = &process_handle {
                let _ = WindowsJob::terminate_process_handle(process_handle);
            }
            let _ = child.start_kill();
            return Err(error).context("failed to contain Windows child process");
        }
        Ok(ManagedChild {
            child: Some(child),
            containment: Self { job: Some(job) },
        })
    }

    fn terminate(&mut self) -> bool {
        self.job.as_ref().is_some_and(|job| job.terminate().is_ok())
    }

    fn terminate_gracefully(&mut self) {
        self.terminate();
    }

    fn finish_normally(&mut self) -> std::io::Result<()> {
        if let Some(job) = self.job.take() {
            job.preserve_descendants()
        } else {
            Ok(())
        }
    }
}

#[cfg(windows)]
struct WindowsJob {
    handle: std::os::windows::io::OwnedHandle,
}

#[cfg(windows)]
impl WindowsJob {
    fn create() -> std::io::Result<Self> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle};
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::JobObjects::{
            CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if raw.is_null() || raw == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        let handle = unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(raw.cast()) };
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = unsafe {
            SetInformationJobObject(
                handle.as_raw_handle().cast(),
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { handle })
    }

    fn open_process_handle(process_id: u32) -> std::io::Result<std::os::windows::io::OwnedHandle> {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_TERMINATE};
        let raw = unsafe { OpenProcess(PROCESS_TERMINATE, 0, process_id) };
        if raw.is_null() {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(raw.cast()) })
        }
    }

    fn terminate_process_handle(handle: &std::os::windows::io::OwnedHandle) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::Threading::TerminateProcess;
        let ok = unsafe { TerminateProcess(handle.as_raw_handle().cast(), 1) };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn assign_and_resume(&self, process_id: u32) -> std::io::Result<()> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle};
        use windows_sys::Win32::System::JobObjects::AssignProcessToJobObject;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SET_QUOTA, PROCESS_SUSPEND_RESUME, PROCESS_TERMINATE,
        };

        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn NtResumeProcess(process_handle: *mut core::ffi::c_void) -> i32;
        }

        let raw = unsafe {
            OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_SUSPEND_RESUME,
                0,
                process_id,
            )
        };
        if raw.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let process = unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(raw.cast()) };
        let assigned = unsafe {
            AssignProcessToJobObject(
                self.handle.as_raw_handle().cast(),
                process.as_raw_handle().cast(),
            )
        };
        if assigned == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let status = unsafe { NtResumeProcess(process.as_raw_handle().cast()) };
        if status < 0 {
            let _ = Self::terminate_process_handle(&process);
            return Err(std::io::Error::other(format!(
                "failed to resume Windows process: NTSTATUS {status:#x}"
            )));
        }
        Ok(())
    }

    fn terminate(&self) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;
        let ok = unsafe { TerminateJobObject(self.handle.as_raw_handle().cast(), 1) };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn preserve_descendants(self) -> std::io::Result<()> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::System::JobObjects::{
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };
        let limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        let ok = unsafe {
            SetInformationJobObject(
                self.handle.as_raw_handle().cast(),
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(not(any(unix, windows)))]
struct Containment;

#[cfg(not(any(unix, windows)))]
impl Containment {
    fn spawn(mut command: Command) -> Result<ManagedChild> {
        let child = command.spawn()?;
        Ok(ManagedChild {
            child: Some(child),
            containment: Self,
        })
    }

    fn terminate(&mut self) -> bool {
        false
    }

    fn terminate_gracefully(&mut self) {}

    fn finish_normally(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
