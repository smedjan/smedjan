//! On-policy self-distillation with textual feedback — smedjan's adaptation of Cursor's
//! Composer 2.5 technique for localized credit assignment during fine-tuning.
//!
//! The idea: instead of a single noisy reward over a 100k-token rollout, insert a *localized
//! hint* at the specific turn where the model could have done better → creates a "teacher"
//! distribution → KL-distill the student (original context) toward the teacher. This gives
//! precise, per-turn credit assignment without needing a full RL infrastructure.
//!
//! For cyber fine-tuning: when the model gives a wrong CVE or wrong hashcat mode, insert the
//! correct answer as a hint at that turn → teacher → distill. The model learns the specific
//! correction, not just "this rollout was bad."
//!
//! Implementation: two forward passes (teacher with hint, student without), KL-divergence
//! loss between their token distributions at the target turn, plus the standard cross-entropy
//! loss on the target tokens.

use crate::gpu::MetalContext;
use crate::tensor::Tensor;
use std::sync::Arc;

/// A textual feedback sample: the original context, the hint to insert, and which turn to
/// apply it at. The teacher forward uses `context + hint inserted at turn_idx`; the student
/// forward uses the original `context`.
pub struct TextualFeedbackSample {
    /// Token IDs of the full conversation/context.
    pub token_ids: Vec<u32>,
    /// The target turn index where the hint should be inserted (for the teacher).
    pub turn_idx: usize,
    /// The hint text to insert (tokenized).
    pub hint_token_ids: Vec<u32>,
    /// The correct target tokens for the loss (what the model should have said at turn_idx).
    pub target_token_ids: Vec<u32>,
}

/// Build a teacher input by inserting the hint tokens at the specified turn index.
/// The teacher sees `token_ids[0..turn_idx] + hint_token_ids + token_ids[turn_idx..]`,
/// while the student sees the original `token_ids`. This creates a "what if the model
/// had known the correct answer at this turn" distribution for KL distillation.
pub fn build_teacher_input_from_sample(sample: &TextualFeedbackSample, seq_len: usize) -> Vec<u32> {
    let mut teacher = Vec::with_capacity(seq_len);
    let ctx_end = sample.turn_idx.min(sample.token_ids.len()).min(seq_len);
    teacher.extend_from_slice(&sample.token_ids[..ctx_end]);
    let remaining = seq_len.saturating_sub(teacher.len());
    let hint_len = sample.hint_token_ids.len().min(remaining);
    teacher.extend_from_slice(&sample.hint_token_ids[..hint_len]);
    let remaining = seq_len.saturating_sub(teacher.len());
    if remaining > 0 && ctx_end < sample.token_ids.len() {
        let after = &sample.token_ids[ctx_end..];
        teacher.extend_from_slice(&after[..after.len().min(remaining)]);
    }
    teacher.resize(seq_len, 0);
    teacher
}

/// KL-divergence loss between teacher and student distributions: `KL(teacher || student)`.
/// This is the self-distillation loss — it pushes the student's probabilities toward the
/// teacher's (which incorporates the hint).
///
/// Implementation: `KL = sum_i teacher_i * (log(teacher_i) - log(student_i))`.
/// Both inputs are log-probabilities (log-softmax outputs).
pub fn kl_div_loss(
    ctx: &Arc<MetalContext>,
    teacher_logits: &Tensor, // [seq, vocab]
    student_logits: &Tensor, // [seq, vocab]
    target_mask: &[bool],    // [seq] — only compute KL at the target turn(s)
) -> Tensor {
    let (seq, vocab) = (teacher_logits.shape[0], teacher_logits.shape[1]);
    assert_eq!(student_logits.shape, vec![seq, vocab]);

    // Compute log-softmax for both (the autograd-aware version).
    let teacher_logp = teacher_logits.log_softmax();
    let student_logp = student_logits.log_softmax();

    // KL(teacher || student) = sum(teacher * (teacher_logp - student_logp))
    let teacher_p = teacher_logp.exp();
    let diff = teacher_logp.add(&student_logp.scale(-1.0));
    let kl = teacher_p.mul(&diff);

    // Mask: zero out non-target positions. Expand [seq] mask to [seq, vocab]
    // by repeating each mask value across the vocab dimension.
    let mask_vals: Vec<f32> = target_mask
        .iter()
        .flat_map(|&b| std::iter::repeat_n(if b { 1.0 } else { 0.0 }, vocab))
        .collect();
    let mask = Tensor::from_slice(ctx, &mask_vals, vec![seq, vocab]);
    let kl_masked = kl.mul(&mask);

    let n_targets = target_mask.iter().filter(|&&b| b).count().max(1);
    kl_masked.sum_all().scale(1.0 / n_targets as f32)
}

/// Combined loss for one textual-feedback training step:
///   `loss = CE(student, targets) + alpha * KL(teacher, student)`
/// The CE loss trains the student to produce the correct tokens; the KL loss pushes the
/// student's distribution toward the teacher's (which incorporates the hint). `alpha` controls
/// the strength of the distillation (typically 0.5–1.0).
///
/// Returns `(loss_tensor, ce_grad_buffer)` — the CE gradient buffer is returned so the
/// caller can apply a loss mask (zero out prompt-position gradients) before backward.
pub fn textual_feedback_loss(
    ctx: &Arc<MetalContext>,
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    targets: &[u32],
    target_mask: &[bool],
    alpha: f32,
) -> (Tensor, crate::gpu::Buf) {
    let (ce, ce_grad) = crate::loss::cross_entropy_loss(ctx, student_logits, targets);
    let kl = kl_div_loss(ctx, teacher_logits, student_logits, target_mask);
    (ce.add(&kl.scale(alpha)), ce_grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn textual_feedback_sample_and_loss_finite() {
        let ctx = MetalContext::new();
        let seq = 4;
        let vocab = 8;
        let logits = Tensor::randn(&ctx, vec![seq, vocab], 0.1);
        let teacher = Tensor::randn(&ctx, vec![seq, vocab], 0.1);
        let targets: Vec<u32> = (0..seq).map(|i| (i % vocab) as u32).collect();
        let mask = vec![true; seq];

        let sample = TextualFeedbackSample {
            token_ids: (0..seq).map(|i| i as u32).collect(),
            turn_idx: 1,
            hint_token_ids: vec![0],
            target_token_ids: targets.clone(),
        };
        assert_eq!(sample.target_token_ids.len(), seq);

        let (loss, grad) = textual_feedback_loss(&ctx, &logits, &teacher, &targets, &mask, 0.5);
        let loss_val = loss.to_vec()[0];
        assert!(
            loss_val.is_finite(),
            "loss should be finite, got {loss_val}"
        );
        assert!(
            crate::gpu::buf_len_bytes(&grad) > 0,
            "gradient buffer should be non-empty"
        );

        // Verify build_teacher_input_from_sample produces correct length.
        let teacher_input = build_teacher_input_from_sample(&sample, seq);
        assert_eq!(teacher_input.len(), seq);
    }
}
