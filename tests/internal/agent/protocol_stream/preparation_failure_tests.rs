use super::*;

#[test]
fn preparation_preserves_typed_model_failure_through_context() {
    let failure = ModelFailure::new(
        crate::model_runtime::FailurePhase::Transport,
        crate::model_runtime::FailureKind::Http,
    )
    .with_status(200)
    .with_code("response_chunk_failed")
    .with_retry_hint(crate::model_runtime::RetryHint::Retryable)
    .with_detail("summary stream interrupted");
    let error = anyhow::Error::new(failure.clone()).context("summary request");
    assert_eq!(preparation_failure(error), failure);
    let local = preparation_failure(anyhow!("invalid local history"));
    assert_eq!(local.phase, crate::model_runtime::FailurePhase::Prepare);
    assert_eq!(local.kind, crate::model_runtime::FailureKind::Internal);
    assert_eq!(local.retry_hint, crate::model_runtime::RetryHint::Never);
}
