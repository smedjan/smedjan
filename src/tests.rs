#[cfg(test)]
mod tests {
    use crate::autograd;
    use crate::datapipe;
    use crate::metal::MetalContext;
    use crate::model::ModelConfig;
    use crate::tensor::Tensor;
    use crate::tokenizer::{BpeTokenizer, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SPECIAL_TOKENS};
    use std::sync::Arc;

    /// Helper: create a MetalContext for tests. Panics if no GPU available.
    fn test_ctx() -> Arc<MetalContext> {
        MetalContext::new()
    }

    // =========================================================================
    // 1. Tensor operations
    // =========================================================================

    #[test]
    fn tensor_from_slice_roundtrip() {
        let ctx = test_ctx();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&ctx, &data, vec![2, 3]);
        let readback = t.to_vec();
        assert_eq!(readback, data);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn tensor_add() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let b = Tensor::from_slice(&ctx, &[10.0, 20.0, 30.0, 40.0], vec![2, 2]);
            let c = a.add(&b);
            let result = c.to_vec();
            assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
            assert_eq!(c.shape, vec![2, 2]);
        });
    }

    #[test]
    fn tensor_mul() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let b = Tensor::from_slice(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]);
            let c = a.mul(&b);
            let result = c.to_vec();
            assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
            assert_eq!(c.shape, vec![2, 2]);
        });
    }

    #[test]
    fn tensor_matmul_2x3_times_3x4() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // A = [[1, 2, 3],
            //      [4, 5, 6]]  shape [2, 3]
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

            // B = [[1, 2, 3, 4],
            //      [5, 6, 7, 8],
            //      [9, 10, 11, 12]]  shape [3, 4]
            let b = Tensor::from_slice(
                &ctx,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                vec![3, 4],
            );

            let c = a.matmul(&b);
            assert_eq!(c.shape, vec![2, 4]);

            let result = c.to_vec();
            // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
            //       = [38, 44, 50, 56]
            // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
            //       = [83, 98, 113, 128]
            let expected = vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
            for (got, exp) in result.iter().zip(expected.iter()) {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "matmul mismatch: got {}, expected {}",
                    got,
                    exp
                );
            }
        });
    }

    #[test]
    fn tensor_scale() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let a = Tensor::from_slice(&ctx, &[2.0, 4.0, 6.0], vec![1, 3]);
            let b = a.scale(0.5);
            let result = b.to_vec();
            assert_eq!(result, vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn tensor_reshape() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let a = Tensor::from_slice(&ctx, &data, vec![2, 3]);
            let b = a.reshape(vec![3, 2]);

            assert_eq!(b.shape, vec![3, 2]);
            // Data is preserved (reshape is a view, no copy)
            assert_eq!(b.to_vec(), data);
            assert_eq!(b.numel(), 6);
        });
    }

    #[test]
    fn tensor_softmax_rows_sum_to_one() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // 2 rows x 4 cols — each row should sum to ~1.0 after softmax
            let a = Tensor::from_slice(
                &ctx,
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![2, 4],
            );
            let s = a.softmax();
            let result = s.to_vec();
            assert_eq!(result.len(), 8);

            // Check row 0 sums to ~1.0
            let row0_sum: f32 = result[0..4].iter().sum();
            assert!(
                (row0_sum - 1.0).abs() < 1e-5,
                "Row 0 sum: {}, expected ~1.0",
                row0_sum
            );

            // Check row 1 sums to ~1.0
            let row1_sum: f32 = result[4..8].iter().sum();
            assert!(
                (row1_sum - 1.0).abs() < 1e-5,
                "Row 1 sum: {}, expected ~1.0",
                row1_sum
            );

            // All values should be positive
            for val in &result {
                assert!(*val > 0.0, "Softmax output should be positive, got {}", val);
            }

            // Values should be monotonically increasing within each row (since inputs are)
            for row in 0..2 {
                for col in 1..4 {
                    let prev = result[row * 4 + col - 1];
                    let curr = result[row * 4 + col];
                    assert!(
                        curr >= prev,
                        "Softmax should preserve ordering: {} >= {}",
                        curr,
                        prev
                    );
                }
            }
        });
    }

    #[test]
    fn tensor_zeros_and_full() {
        let ctx = test_ctx();
        let z = Tensor::zeros(&ctx, vec![3, 4]);
        assert_eq!(z.shape, vec![3, 4]);
        assert!(z.to_vec().iter().all(|&v| v == 0.0));

        let f = Tensor::full(&ctx, vec![2, 3], 7.5);
        assert_eq!(f.shape, vec![2, 3]);
        assert!(f.to_vec().iter().all(|&v| (v - 7.5).abs() < 1e-6));
    }

    #[test]
    fn tensor_with_grad() {
        let ctx = test_ctx();
        let a = Tensor::from_slice(&ctx, &[1.0, 2.0], vec![1, 2]);
        assert!(!a.requires_grad);
        let b = a.with_grad();
        assert!(b.requires_grad);
    }

    #[test]
    fn tensor_detach() {
        let ctx = test_ctx();
        let a = Tensor::from_slice(&ctx, &[1.0, 2.0], vec![1, 2]).with_grad();
        assert!(a.requires_grad);
        let b = a.detach();
        assert!(!b.requires_grad);
        // Data is shared
        assert_eq!(b.to_vec(), vec![1.0, 2.0]);
    }

    // =========================================================================
    // 2. Autograd
    // =========================================================================

    #[test]
    fn autograd_matmul_backward_produces_gradients() {
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        // Create parameter tensors (requires_grad = true)
        let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).with_grad();
        let b = Tensor::from_slice(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]).with_grad();

        let c = a.matmul(&b);

        // Create a simple scalar loss: sum of all output elements via scale
        // c is [2,2], sum it by reading values
        let c_data = c.to_vec();
        let loss_val: f32 = c_data.iter().sum();
        assert!(loss_val > 0.0, "Matmul output should be non-zero");

        // Backward from the matmul output
        autograd::backward(&ctx, c.id);

        // Both inputs should have gradients
        let grad_a = autograd::get_grad(a.id);
        let grad_b = autograd::get_grad(b.id);

        assert!(grad_a.is_some(), "Gradient for A should exist after backward");
        assert!(grad_b.is_some(), "Gradient for B should exist after backward");

        // Gradients should be non-zero
        let grad_a_data = MetalContext::read_buffer(&grad_a.unwrap(), 4);
        let grad_b_data = MetalContext::read_buffer(&grad_b.unwrap(), 4);

        let grad_a_norm: f32 = grad_a_data.iter().map(|x| x * x).sum();
        let grad_b_norm: f32 = grad_b_data.iter().map(|x| x * x).sum();

        assert!(
            grad_a_norm > 0.0,
            "Gradient for A should be non-zero, got L2={}",
            grad_a_norm
        );
        assert!(
            grad_b_norm > 0.0,
            "Gradient for B should be non-zero, got L2={}",
            grad_b_norm
        );

        autograd::clear_tape();
    }

    #[test]
    fn autograd_add_backward_both_inputs_get_gradients() {
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        // Use 1x1 tensors to match the scalar loss gradient that backward() seeds.
        // backward() initializes a single 1.0 scalar as the output gradient, so the
        // output must be 1 element for the gradient to propagate correctly through add.
        let a = Tensor::from_slice(&ctx, &[3.0], vec![1, 1]).with_grad();
        let b = Tensor::from_slice(&ctx, &[7.0], vec![1, 1]).with_grad();

        let c = a.add(&b);

        autograd::backward(&ctx, c.id);

        let grad_a = autograd::get_grad(a.id);
        let grad_b = autograd::get_grad(b.id);

        assert!(grad_a.is_some(), "Gradient for A should exist after add backward");
        assert!(grad_b.is_some(), "Gradient for B should exist after add backward");

        // For scalar addition, both gradients should be 1.0 (pass-through from upstream)
        let grad_a_data = MetalContext::read_buffer(&grad_a.unwrap(), 1);
        let grad_b_data = MetalContext::read_buffer(&grad_b.unwrap(), 1);

        assert!(
            (grad_a_data[0] - 1.0).abs() < 1e-5,
            "Add grad_a should be 1.0, got {}",
            grad_a_data[0]
        );
        assert!(
            (grad_b_data[0] - 1.0).abs() < 1e-5,
            "Add grad_b should be 1.0, got {}",
            grad_b_data[0]
        );

        autograd::clear_tape();
    }

    #[test]
    fn autograd_clear_tape_empties_tape() {
        let ctx = test_ctx();
        autograd::clear_tape();

        // Record some ops
        let a = Tensor::from_slice(&ctx, &[1.0, 2.0], vec![1, 2]).with_grad();
        let b = Tensor::from_slice(&ctx, &[3.0, 4.0], vec![1, 2]).with_grad();
        let _c = a.add(&b);

        // Tape should have entries
        let (num_ops, _) = autograd::tape_stats();
        assert!(num_ops > 0, "Tape should have ops recorded");

        // Clear
        autograd::clear_tape();

        let (num_ops_after, _) = autograd::tape_stats();
        assert_eq!(num_ops_after, 0, "Tape should be empty after clear");
    }

    #[test]
    fn autograd_no_grad_prevents_recording() {
        let ctx = test_ctx();
        autograd::clear_tape();

        autograd::no_grad(|| {
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0], vec![1, 2]);
            let b = Tensor::from_slice(&ctx, &[3.0, 4.0], vec![1, 2]);
            let _c = a.add(&b);
        });

        let (num_ops, _) = autograd::tape_stats();
        assert_eq!(num_ops, 0, "No ops should be recorded inside no_grad");
    }

    // =========================================================================
    // 3. Tokenizer
    // =========================================================================

    #[test]
    fn tokenizer_encode_decode_roundtrip() {
        // Train a small tokenizer on a simple corpus
        let corpus = b"hello world hello world the quick brown fox jumps over the lazy dog. \
                       The quick brown fox jumps over the lazy dog again and again.";
        let tok = BpeTokenizer::train(corpus, 300);

        let text = "hello world";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);

        assert_eq!(decoded, text, "Encode/decode roundtrip should preserve text");
    }

    #[test]
    fn tokenizer_special_token_ids() {
        assert_eq!(PAD_TOKEN, 0, "PAD_TOKEN should be 0");
        assert_eq!(BOS_TOKEN, 1, "BOS_TOKEN should be 1");
        assert_eq!(EOS_TOKEN, 2, "EOS_TOKEN should be 2");
        assert_eq!(SPECIAL_TOKENS, 3, "SPECIAL_TOKENS count should be 3");
    }

    #[test]
    fn tokenizer_vocab_size_after_training() {
        let corpus = b"abcabcabcabc def def def ghi ghi jkl";
        let target_vocab = 280; // 256 bytes + 3 special + some merges
        let tok = BpeTokenizer::train(corpus, target_vocab);

        let vocab_size = tok.vocab_size();
        // Vocab should be between (256 + 3) and target
        assert!(
            vocab_size >= 259,
            "Vocab should have at least 259 entries (256 bytes + 3 special), got {}",
            vocab_size
        );
        assert!(
            vocab_size <= target_vocab,
            "Vocab should not exceed target {}, got {}",
            target_vocab,
            vocab_size
        );

        // Verify merge count matches
        let expected_merges = vocab_size - 259;
        assert_eq!(
            tok.merges.len() as u32,
            expected_merges,
            "Merge count should equal vocab_size - 259"
        );
    }

    #[test]
    fn tokenizer_contains_known_bytes() {
        let corpus = b"the cat sat on the mat";
        let tok = BpeTokenizer::train(corpus, 280);

        // Single byte tokens should all exist (byte-level BPE)
        for byte in 0u8..=255 {
            let token = vec![byte];
            assert!(
                tok.vocab.contains_key(&token),
                "Byte token {} should exist in vocab",
                byte
            );
        }
    }

    #[test]
    fn tokenizer_encode_produces_valid_ids() {
        let corpus = b"hello world test";
        let tok = BpeTokenizer::train(corpus, 280);

        let encoded = tok.encode("hello");
        let vocab_size = tok.vocab_size();

        for &id in &encoded {
            assert!(
                id < vocab_size,
                "Token ID {} should be < vocab_size {}",
                id,
                vocab_size
            );
        }
        assert!(!encoded.is_empty(), "Encoding should produce at least one token");
    }

    // =========================================================================
    // 4. SHA-256 (datapipe)
    // =========================================================================

    #[test]
    fn sha256_known_string() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let hash = datapipe::sha256(b"hello");
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            "SHA-256 of 'hello' mismatch"
        );
    }

    #[test]
    fn sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = datapipe::sha256(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 of empty string mismatch"
        );
    }

    #[test]
    fn sha256_longer_input() {
        // SHA-256("The quick brown fox jumps over the lazy dog")
        // = d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592
        let hash = datapipe::sha256(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            hash,
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
            "SHA-256 of fox sentence mismatch"
        );
    }

    // =========================================================================
    // 5. Metal context
    // =========================================================================

    #[test]
    fn metal_context_creation_succeeds() {
        let ctx = test_ctx();
        let name = ctx.device_name();
        assert!(!name.is_empty(), "Device name should not be empty");
    }

    #[test]
    fn metal_buffer_alloc_and_readback() {
        let ctx = test_ctx();

        // Allocate buffer from data
        let data = vec![3.14f32, 2.71, 1.41, 1.73];
        let buf = ctx.buffer_from_slice(&data);
        let readback = MetalContext::read_buffer(&buf, 4);
        assert_eq!(readback, data);
    }

    #[test]
    fn metal_buffer_alloc_empty() {
        let ctx = test_ctx();
        // Allocate an empty buffer (at least 4 bytes to avoid zero-size issues)
        let buf = ctx.alloc_buffer(16);
        let readback = MetalContext::read_buffer(&buf, 4);
        // Values are undefined for an uninitialized buffer, but readback should not crash
        assert_eq!(readback.len(), 4);
    }

    #[test]
    fn metal_u32_buffer_roundtrip() {
        let ctx = test_ctx();
        let data = vec![42u32, 100, 0, 999999];
        let buf = ctx.buffer_from_u32_slice(&data);
        let readback = MetalContext::read_buffer_u32(&buf, 4);
        assert_eq!(readback, data);
    }

    // =========================================================================
    // 6. Model config
    // =========================================================================

    #[test]
    fn model_config_tiny_param_count() {
        let cfg = ModelConfig::tiny(8192);
        let params = cfg.param_count();
        assert!(params > 0, "Tiny model should have >0 params, got {}", params);
        assert!(
            params < 10_000_000,
            "Tiny model should have <10M params, got {}",
            params
        );
    }

    #[test]
    fn model_config_sizes_ordered() {
        let v = 8192;
        let tiny = ModelConfig::tiny(v).param_count();
        let small = ModelConfig::small(v).param_count();
        let medium = ModelConfig::medium(v).param_count();
        let large = ModelConfig::large(v).param_count();

        assert!(
            tiny < small,
            "tiny ({}) should have fewer params than small ({})",
            tiny,
            small
        );
        assert!(
            small < medium,
            "small ({}) should have fewer params than medium ({})",
            small,
            medium
        );
        assert!(
            medium < large,
            "medium ({}) should have fewer params than large ({})",
            medium,
            large
        );
    }

    #[test]
    fn model_config_d_ff_aligned() {
        // d_ff should always be a multiple of 256
        let configs = vec![
            ModelConfig::tiny(8192),
            ModelConfig::small(8192),
            ModelConfig::medium(8192),
            ModelConfig::large(8192),
        ];
        for cfg in &configs {
            let d_ff = cfg.d_ff();
            assert_eq!(
                d_ff % 256,
                0,
                "d_ff ({}) should be aligned to 256 for d_model={}",
                d_ff,
                cfg.d_model
            );
        }
    }

    #[test]
    fn model_config_custom() {
        let cfg = ModelConfig::custom(1000, 64, 4, 2, 2.0, 128);
        assert_eq!(cfg.vocab_size, 1000);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_heads, 4);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.max_seq_len, 128);
        assert!(cfg.param_count() > 0);
    }

    #[test]
    fn model_config_memory_estimates() {
        let cfg = ModelConfig::tiny(8192);
        let train_mem = cfg.training_memory_bytes();
        let infer_mem = cfg.inference_memory_bytes();

        // Training needs more memory than inference (4x: param + grad + m + v)
        assert!(
            train_mem > infer_mem,
            "Training memory ({}) should exceed inference memory ({})",
            train_mem,
            infer_mem
        );
        assert_eq!(
            train_mem,
            infer_mem * 4,
            "Training memory should be exactly 4x inference memory"
        );
    }

    #[test]
    fn model_config_summary_not_empty() {
        let cfg = ModelConfig::small(8192);
        let summary = cfg.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("d_model="));
        assert!(summary.contains("params="));
    }

    // =========================================================================
    // 7. Datapipe quality filter and normalization
    // =========================================================================

    #[test]
    fn datapipe_quality_filter_rejects_short() {
        assert!(!datapipe::quality_filter("too short"), "Should reject short text");
    }

    #[test]
    fn datapipe_quality_filter_accepts_good_text() {
        let good = "This is a good quality document with enough text content. \
                    It has multiple sentences and proper punctuation. \
                    The document contains useful information that would be \
                    suitable for training a language model.";
        assert!(datapipe::quality_filter(good), "Should accept good text");
    }

    #[test]
    fn datapipe_normalize_text_collapses_blanks() {
        let input = "line one\n\n\n\n\nline two\n\n\nline three";
        let normalized = datapipe::normalize_text(input);
        // Multiple blank lines should collapse to single blank line
        assert!(
            !normalized.contains("\n\n\n"),
            "Should not have triple newlines after normalization"
        );
    }

    #[test]
    fn datapipe_exact_dedup() {
        let docs = vec![
            "doc one".to_string(),
            "doc two".to_string(),
            "doc one".to_string(), // duplicate
            "doc three".to_string(),
        ];
        let unique = datapipe::exact_dedup(&docs);
        assert_eq!(unique.len(), 3, "Should remove 1 duplicate");
        assert_eq!(unique, vec![0, 1, 3]);
    }
}
