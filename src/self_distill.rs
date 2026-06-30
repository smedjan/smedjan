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
    // = sum(teacher * teacher_logp) - sum(teacher * student_logp)
    // Since teacher = exp(teacher_logp), this is: sum(teacher_logp.exp() * (teacher_logp - student_logp))
    let teacher_p = teacher_logp.exp();
    let diff = teacher_logp.add(&student_logp.scale(-1.0)); // teacher_logp - student_logp
    let kl = teacher_p.mul(&diff);

    // Mask: zero out non-target positions. The mask is applied by multiplying KL per position
    // by 0 or 1, then summing.
    let mask_vals: Vec<f32> = target_mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
    let mask = Tensor::from_slice(ctx, &mask_vals, vec![seq, 1]);
    let kl_masked = kl.mul(&mask);

    // Sum and normalize by the number of target positions.
    let n_targets = target_mask.iter().filter(|&&b| b).count().max(1);
    kl_masked.sum_all().scale(1.0 / n_targets as f32)
}

/// Combined loss for one textual-feedback training step:
///   `loss = CE(student, targets) + alpha * KL(teacher, student)`
/// The CE loss trains the student to produce the correct tokens; the KL loss pushes the
/// student's distribution toward the teacher's (which incorporates the hint). `alpha` controls
/// the strength of the distillation (typically 0.5–1.0).
pub fn textual_feedback_loss(
    ctx: &Arc<MetalContext>,
    student_logits: &Tensor, // [seq, vocab] — student forward (no hint)
    teacher_logits: &Tensor, // [seq, vocab] — teacher forward (with hint)
    targets: &[u32],         // [n_targets] — correct token IDs
    target_mask: &[bool],    // [seq] — which positions are targets
    alpha: f32,              // KL weight
) -> Tensor {
    // Cross-entropy on the student (standard SFT loss).
    let ce = crate::loss::cross_entropy_loss(ctx, student_logits, targets).0;

    // KL divergence between teacher and student at target positions.
    let kl = kl_div_loss(ctx, teacher_logits, student_logits, target_mask);

    // Combined: CE + alpha * KL. The CE is the primary signal; KL is the localized correction.
    ce.add(&kl.scale(alpha))
}
