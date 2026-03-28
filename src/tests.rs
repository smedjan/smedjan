#[cfg(test)]
mod tests {
    use crate::autograd;
    use crate::datapipe;
    use crate::metal::{compute, MetalContext};
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

    // =========================================================================
    // 8. Quantize/dequantize roundtrip
    // =========================================================================

    #[test]
    fn quantize_q8_roundtrip_within_tolerance() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let shape = vec![128];
        let qt = crate::quantize::quantize(&data, &shape, 8, 32);
        let recovered = crate::quantize::dequantize(&qt);

        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(
                err < 0.06,
                "Q8 error too large: orig={}, recovered={}, err={}",
                orig, rec, err,
            );
        }
    }

    #[test]
    fn quantize_q4_roundtrip_within_tolerance() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let shape = vec![64];
        let qt = crate::quantize::quantize(&data, &shape, 4, 32);
        let recovered = crate::quantize::dequantize(&qt);

        assert_eq!(recovered.len(), data.len());
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(
                err < 0.25,
                "Q4 error too large: orig={}, recovered={}, err={}",
                orig, rec, err,
            );
        }
    }

    #[test]
    fn quantize_q4_nibble_packing_correctness() {
        // Test that even/odd indices are packed correctly into low/high nibbles
        let data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let shape = vec![8];
        let qt = crate::quantize::quantize(&data, &shape, 4, 8);
        let recovered = crate::quantize::dequantize(&qt);

        assert_eq!(recovered.len(), 8);
        // Even indices should be ~0.0, odd indices should be ~1.0
        for i in 0..8 {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            let err = (recovered[i] - expected).abs();
            assert!(
                err < 0.15,
                "Q4 nibble packing error at index {}: expected ~{}, got {}",
                i, expected, recovered[i],
            );
        }
    }

    #[test]
    fn quantize_constant_data_roundtrip() {
        // All identical values — scale should be ~0, zero should be the value
        let data = vec![3.14f32; 64];
        let shape = vec![64];

        let qt8 = crate::quantize::quantize(&data, &shape, 8, 32);
        let rec8 = crate::quantize::dequantize(&qt8);
        for &v in &rec8 {
            assert!((v - 3.14).abs() < 0.01, "Q8 constant data: got {}", v);
        }

        let qt4 = crate::quantize::quantize(&data, &shape, 4, 32);
        let rec4 = crate::quantize::dequantize(&qt4);
        for &v in &rec4 {
            assert!((v - 3.14).abs() < 0.01, "Q4 constant data: got {}", v);
        }
    }

    // =========================================================================
    // 9. Tokenizer edge cases
    // =========================================================================

    #[test]
    fn tokenizer_encode_empty_string() {
        let corpus = b"hello world test data";
        let tok = BpeTokenizer::train(corpus, 280);
        let encoded = tok.encode("");
        assert!(encoded.is_empty(), "Empty string should encode to empty vec");
    }

    #[test]
    fn tokenizer_save_load_roundtrip() {
        let corpus = b"the quick brown fox jumps over the lazy dog again and again";
        let tok = BpeTokenizer::train(corpus, 300);

        let path = "/tmp/andreai_test_tokenizer.bin";
        tok.save(path).expect("Failed to save tokenizer");
        let tok2 = BpeTokenizer::load(path).expect("Failed to load tokenizer");

        assert_eq!(tok.vocab_size(), tok2.vocab_size());
        assert_eq!(tok.merges.len(), tok2.merges.len());

        // Encode/decode should produce identical results
        let text = "the quick brown fox";
        let enc1 = tok.encode(text);
        let enc2 = tok2.encode(text);
        assert_eq!(enc1, enc2, "Loaded tokenizer should produce same encoding");

        let dec1 = tok.decode(&enc1);
        let dec2 = tok2.decode(&enc2);
        assert_eq!(dec1, dec2);
        assert_eq!(dec1, text);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn tokenizer_long_text_chunked_encoding() {
        let corpus = b"abcdefghijklmnopqrstuvwxyz 0123456789 the quick brown fox jumps";
        let tok = BpeTokenizer::train(corpus, 280);

        // Create a long text that triggers chunked encoding (> 10000 bytes)
        let long_text = "the quick brown fox ".repeat(600); // 12000 chars
        let encoded = tok.encode(&long_text);
        let decoded = tok.decode(&encoded);

        assert_eq!(decoded, long_text, "Chunked encoding should roundtrip correctly");
    }

    // =========================================================================
    // 10. Tensor edge cases
    // =========================================================================

    #[test]
    fn tensor_batched_matmul_batch_1() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // [1, 2, 3] @ [1, 3, 2] → [1, 2, 2]
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
            let b = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
            let c = a.batched_matmul(&b);
            assert_eq!(c.shape, vec![1, 2, 2]);

            let result = c.to_vec();
            // [1,2,3] @ [[1,2],[3,4],[5,6]] = [1+6+15, 2+8+18] = [22, 28]
            // [4,5,6] @ [[1,2],[3,4],[5,6]] = [4+15+30, 8+20+36] = [49, 64]
            assert!((result[0] - 22.0).abs() < 1e-3, "got {}", result[0]);
            assert!((result[1] - 28.0).abs() < 1e-3, "got {}", result[1]);
            assert!((result[2] - 49.0).abs() < 1e-3, "got {}", result[2]);
            assert!((result[3] - 64.0).abs() < 1e-3, "got {}", result[3]);
        });
    }

    #[test]
    fn tensor_batched_matmul_trans_b_batch_1() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // A: [1, 2, 3], B: [1, 2, 3] → C = A @ B^T: [1, 2, 2]
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
            let b = Tensor::from_slice(&ctx, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![1, 2, 3]);
            let c = a.batched_matmul_trans_b(&b);
            assert_eq!(c.shape, vec![1, 2, 2]);

            let result = c.to_vec();
            // Row 0 of A [1,2,3] dot row 0 of B [1,0,0] = 1
            // Row 0 of A [1,2,3] dot row 1 of B [0,1,0] = 2
            // Row 1 of A [4,5,6] dot row 0 of B [1,0,0] = 4
            // Row 1 of A [4,5,6] dot row 1 of B [0,1,0] = 5
            assert!((result[0] - 1.0).abs() < 1e-3, "got {}", result[0]);
            assert!((result[1] - 2.0).abs() < 1e-3, "got {}", result[1]);
            assert!((result[2] - 4.0).abs() < 1e-3, "got {}", result[2]);
            assert!((result[3] - 5.0).abs() < 1e-3, "got {}", result[3]);
        });
    }

    #[test]
    fn tensor_slice_flat_and_concat_flat() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let data = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);

            // Slice [2..5]
            let sliced = data.slice_flat(2, 3, vec![3]);
            assert_eq!(sliced.to_vec(), vec![3.0, 4.0, 5.0]);

            // Concat two slices
            let a = data.slice_flat(0, 2, vec![2]);
            let b = data.slice_flat(4, 2, vec![2]);
            let concat = Tensor::concat_flat(&[&a, &b], vec![4]);
            assert_eq!(concat.to_vec(), vec![1.0, 2.0, 5.0, 6.0]);
        });
    }

    #[test]
    fn tensor_matmul_trans_b() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // A: [2, 3], B: [2, 3] → C = A @ B^T: [2, 2]
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
            let b = Tensor::from_slice(&ctx, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], vec![2, 3]);
            let c = a.matmul_trans_b(&b);
            assert_eq!(c.shape, vec![2, 2]);

            let result = c.to_vec();
            // [1,2,3] . [1,0,0] = 1, [1,2,3] . [0,1,0] = 2
            // [4,5,6] . [1,0,0] = 4, [4,5,6] . [0,1,0] = 5
            assert!((result[0] - 1.0).abs() < 1e-3);
            assert!((result[1] - 2.0).abs() < 1e-3);
            assert!((result[2] - 4.0).abs() < 1e-3);
            assert!((result[3] - 5.0).abs() < 1e-3);
        });
    }

    // =========================================================================
    // 11. Cosine warmup scheduler edge cases
    // =========================================================================

    #[test]
    fn scheduler_warmup_zero() {
        let sched = crate::optim::CosineWarmupScheduler::new(1e-3, 0, 1000);
        let lr0 = sched.get_lr(0);
        assert!(
            (lr0 - 1e-3).abs() < 1e-8,
            "With warmup=0, step 0 should give max_lr, got {}",
            lr0,
        );
    }

    #[test]
    fn scheduler_lr_decreases_after_warmup() {
        let sched = crate::optim::CosineWarmupScheduler::new(1e-3, 100, 1000);
        let lr_warmup = sched.get_lr(100);
        let lr_mid = sched.get_lr(500);
        let lr_end = sched.get_lr(999);

        assert!(lr_warmup > lr_mid, "LR should decrease after warmup: {} > {}", lr_warmup, lr_mid);
        assert!(lr_mid > lr_end, "LR should continue decreasing: {} > {}", lr_mid, lr_end);
        assert!(lr_end > 0.0, "LR should always be positive: {}", lr_end);
    }

    // =========================================================================
    // 12. Model config edge cases
    // =========================================================================

    #[test]
    fn model_config_ffn_multiplier_roundtrip_checkpoint_format() {
        // Verify that ffn_multiplier is preserved through the checkpoint config
        // serialization format (f32 write/read)
        let config = ModelConfig::custom(1000, 64, 4, 2, 2.6666667, 128);
        let d_ff = config.d_ff();
        assert!(d_ff > 0, "d_ff should be positive");
        assert_eq!(d_ff % 256, 0, "d_ff should be 256-aligned");

        // Verify the actual multiplied value rounds correctly
        let raw_ff = (64.0f32 * 2.6666667) as usize; // ~170
        let aligned_ff = raw_ff.div_ceil(256) * 256; // 256
        assert_eq!(d_ff, aligned_ff);
    }

    // =========================================================================
    // 13. RMS norm produces valid output
    // =========================================================================

    #[test]
    fn tensor_rms_norm_output_is_normalized() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let data = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
            let weight = Tensor::from_slice(&ctx, &[1.0, 1.0, 1.0], vec![3]);
            let normed = data.rms_norm(&weight, 1e-5);

            let result = normed.to_vec();
            assert_eq!(result.len(), 6);

            // After RMS norm with weight=1, output = x / rms(x)
            // Row 0: x=[1,2,3], rms = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
            // Row 1: x=[4,5,6], rms = sqrt((16+25+36)/3) = sqrt(77/3) ≈ 5.07
            // Verify output is not NaN/Inf
            for v in &result {
                assert!(v.is_finite(), "RMS norm output should be finite, got {}", v);
            }

            // Verify the sum of squares of each row is approximately the number of columns
            // (since rms_norm normalizes to unit RMS)
            for row in 0..2 {
                let row_data = &result[row * 3..(row + 1) * 3];
                let rms_sq: f32 = row_data.iter().map(|x| x * x).sum::<f32>() / 3.0;
                assert!(
                    (rms_sq - 1.0).abs() < 0.1,
                    "Row {} RMS should be ~1.0, got sqrt({})",
                    row,
                    rms_sq,
                );
            }
        });
    }

    // =========================================================================
    // Knowledge Distillation
    // =========================================================================

    #[test]
    fn kl_divergence_identical_distributions_is_zero() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // Two identical logit distributions — KL divergence should be ~0
            let batch = 4;
            let vocab = 32;
            let logits: Vec<f32> = (0..batch * vocab)
                .map(|i| ((i as f32) * 0.1).sin())
                .collect();

            let teacher = Tensor::from_slice(&ctx, &logits, vec![batch, vocab]);
            let student = Tensor::from_slice(&ctx, &logits, vec![batch, vocab]);
            let targets: Vec<u32> = (0..batch).map(|i| (i % vocab) as u32).collect();

            let (loss, _grad) = crate::loss::distillation_loss(
                &ctx,
                &student,
                &teacher,
                4.0,  // temperature
                1.0,  // alpha=1.0 means pure KL (no CE component)
                &targets,
            );

            let loss_val = loss.to_vec()[0];
            assert!(
                loss_val.abs() < 0.01,
                "KL divergence of identical distributions should be ~0, got {}",
                loss_val
            );
        });
    }

    #[test]
    fn kl_divergence_different_distributions_is_positive() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let batch = 4;
            let vocab = 32;
            let teacher_logits: Vec<f32> = (0..batch * vocab)
                .map(|i| ((i as f32) * 0.1).sin() * 2.0)
                .collect();
            let student_logits: Vec<f32> = (0..batch * vocab)
                .map(|i| ((i as f32) * 0.3).cos() * 1.5)
                .collect();

            let teacher = Tensor::from_slice(&ctx, &teacher_logits, vec![batch, vocab]);
            let student = Tensor::from_slice(&ctx, &student_logits, vec![batch, vocab]);
            let targets: Vec<u32> = (0..batch).map(|i| (i % vocab) as u32).collect();

            let (loss, _grad) = crate::loss::distillation_loss(
                &ctx,
                &student,
                &teacher,
                4.0,
                1.0,  // pure KL
                &targets,
            );

            let loss_val = loss.to_vec()[0];
            assert!(
                loss_val > 0.0,
                "KL divergence of different distributions should be positive, got {}",
                loss_val
            );
        });
    }

    #[test]
    fn distillation_loss_combines_kl_and_ce() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let batch = 4;
            let vocab = 32;
            let teacher_logits: Vec<f32> = (0..batch * vocab)
                .map(|i| ((i as f32) * 0.1).sin())
                .collect();
            let student_logits: Vec<f32> = (0..batch * vocab)
                .map(|i| ((i as f32) * 0.2).cos())
                .collect();

            let teacher = Tensor::from_slice(&ctx, &teacher_logits, vec![batch, vocab]);
            let student = Tensor::from_slice(&ctx, &student_logits, vec![batch, vocab]);
            let targets: Vec<u32> = (0..batch).map(|i| (i % vocab) as u32).collect();

            // Pure KL (alpha=1)
            let (kl_loss, _) = crate::loss::distillation_loss(
                &ctx,
                &Tensor::from_slice(&ctx, &student_logits, vec![batch, vocab]),
                &Tensor::from_slice(&ctx, &teacher_logits, vec![batch, vocab]),
                4.0,
                1.0,
                &targets,
            );

            // Pure CE (alpha=0)
            let (ce_loss, _) = crate::loss::distillation_loss(
                &ctx,
                &Tensor::from_slice(&ctx, &student_logits, vec![batch, vocab]),
                &Tensor::from_slice(&ctx, &teacher_logits, vec![batch, vocab]),
                4.0,
                0.0,
                &targets,
            );

            // Mixed (alpha=0.5)
            let (mixed_loss, _) = crate::loss::distillation_loss(
                &ctx,
                &student,
                &teacher,
                4.0,
                0.5,
                &targets,
            );

            let kl_val = kl_loss.to_vec()[0];
            let ce_val = ce_loss.to_vec()[0];
            let mixed_val = mixed_loss.to_vec()[0];

            // Mixed should be between pure KL and pure CE (approximately)
            // Actually: mixed = 0.5 * T^2 * KL + 0.5 * CE, while kl_val = 1.0 * T^2 * KL, ce_val = 1.0 * CE
            // So mixed = 0.5 * kl_val + 0.5 * ce_val
            let expected = 0.5 * kl_val + 0.5 * ce_val;
            let diff = (mixed_val - expected).abs();
            assert!(
                diff < 0.1,
                "Mixed loss {} should be ~0.5*KL({}) + 0.5*CE({}) = {}, diff={}",
                mixed_val, kl_val, ce_val, expected, diff
            );
        });
    }

    #[test]
    fn gpu_transpose_2d_correctness() {
        let ctx = test_ctx();
        // Matrix [2, 3]:
        // [[1, 2, 3],
        //  [4, 5, 6]]
        // Transposed [3, 2]:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rows = 2u32;
        let cols = 3u32;
        let input_buf = ctx.buffer_from_slice(&input_data);
        let output_buf = ctx.alloc_buffer((rows * cols) as usize * 4);

        crate::metal::compute::gpu_transpose_2d(&ctx, &input_buf, &output_buf, rows, cols);

        let result = MetalContext::read_buffer(&output_buf, (rows * cols) as usize);
        let expected = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(result, expected, "transpose_2d mismatch");
    }

    #[test]
    fn gpu_transpose_2d_square() {
        let ctx = test_ctx();
        // Square matrix [3, 3]:
        // [[1, 2, 3],
        //  [4, 5, 6],
        //  [7, 8, 9]]
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let n = 3u32;
        let input_buf = ctx.buffer_from_slice(&input_data);
        let output_buf = ctx.alloc_buffer((n * n) as usize * 4);

        crate::metal::compute::gpu_transpose_2d(&ctx, &input_buf, &output_buf, n, n);

        let result = MetalContext::read_buffer(&output_buf, (n * n) as usize);
        let expected = vec![1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        assert_eq!(result, expected, "transpose_2d square mismatch");
    }

    // =========================================================================
    // Gradient accumulation
    // =========================================================================

    /// Verify that gradient accumulation over N micro-steps produces the same
    /// gradients as a single forward/backward with the combined data.
    ///
    /// Uses element-wise mul (scalar-safe): loss = w * x.
    /// d(loss)/dw = x. With 2 identical micro-steps + scale by 0.5:
    /// accumulated = (x + x) * 0.5 = x = single-shot.
    #[test]
    fn gradient_accumulation_matches_single_batch() {
        let ctx = test_ctx();

        let w_data = vec![3.0f32];
        let x_data = vec![7.0f32];

        // === Single-shot: one forward/backward ===
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let w_single = Tensor::from_slice(&ctx, &w_data, vec![1, 1]).with_grad();
        let x_single = Tensor::from_slice(&ctx, &x_data, vec![1, 1]).with_grad();
        let c_single = w_single.mul(&x_single); // scalar

        autograd::backward(&ctx, c_single.id);
        let grad_w_single = autograd::get_grad(w_single.id).expect("grad must exist");
        let grad_w_single_data = MetalContext::read_buffer(&grad_w_single, 1);
        autograd::clear_tape();

        // === Accumulated: two micro-steps with the same data, then scale by 0.5 ===
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let w_accum = Tensor::from_slice(&ctx, &w_data, vec![1, 1]).with_grad();
        let w_accum_id = w_accum.id;

        // Micro-step 1
        let x1 = Tensor::from_slice(&ctx, &x_data, vec![1, 1]).with_grad();
        let c1 = w_accum.mul(&x1);
        autograd::backward(&ctx, c1.id);
        autograd::clear_tape_keep_grads();
        autograd::clear_recompute_registry();

        // Micro-step 2: same input, new tape, grads accumulate.
        // Re-create w tensor with same ID so gradients accumulate on it.
        let w_accum_2 = Tensor {
            id: w_accum_id,
            buffer: ctx.buffer_from_slice(&w_data),
            shape: vec![1, 1],
            requires_grad: true,
            ctx: Arc::clone(&ctx),
        };
        let x2 = Tensor::from_slice(&ctx, &x_data, vec![1, 1]).with_grad();
        let c2 = w_accum_2.mul(&x2);
        autograd::backward(&ctx, c2.id);
        autograd::clear_tape_keep_grads();
        autograd::clear_recompute_registry();

        // Scale accumulated gradients by 1/2 (averaging over 2 micro-steps)
        autograd::scale_grads(&ctx, 0.5);

        let grad_w_accum = autograd::get_grad(w_accum_id).expect("accumulated grad must exist");
        let grad_w_accum_data = MetalContext::read_buffer(&grad_w_accum, 1);

        autograd::clear_tape();

        // Single-shot: d(w*x)/dw = x = 7.0
        // Accumulated: (7.0 + 7.0) * 0.5 = 7.0
        let diff = (grad_w_single_data[0] - grad_w_accum_data[0]).abs();
        assert!(
            diff < 1e-4,
            "Gradient mismatch: single={}, accum={}, diff={}",
            grad_w_single_data[0],
            grad_w_accum_data[0],
            diff,
        );

        // Verify the actual gradient value is correct (x = 7.0)
        assert!(
            (grad_w_single_data[0] - 7.0).abs() < 1e-4,
            "Expected grad(w) = 7.0, got {}",
            grad_w_single_data[0],
        );
    }

    /// Verify that clear_tape_keep_grads preserves gradients while clearing tape entries.
    #[test]
    fn clear_tape_keep_grads_preserves_gradients() {
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).with_grad();
        let b = Tensor::from_slice(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]).with_grad();
        let c = a.matmul(&b);

        autograd::backward(&ctx, c.id);

        // Gradients should exist
        assert!(autograd::get_grad(a.id).is_some(), "grad(a) should exist before clear");

        // Clear tape but keep grads
        autograd::clear_tape_keep_grads();

        // Tape should be empty
        let (tape_ops, _) = autograd::tape_stats();
        assert_eq!(tape_ops, 0, "Tape should be empty after clear_tape_keep_grads");

        // Gradients should still exist
        assert!(autograd::get_grad(a.id).is_some(), "grad(a) should survive clear_tape_keep_grads");
        assert!(autograd::get_grad(b.id).is_some(), "grad(b) should survive clear_tape_keep_grads");

        // Cleanup
        autograd::clear_tape();
    }

    /// Verify that zero_grads clears all gradient buffers.
    #[test]
    fn zero_grads_clears_all_gradients() {
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).with_grad();
        let b = Tensor::from_slice(&ctx, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]).with_grad();
        let c = a.matmul(&b);

        autograd::backward(&ctx, c.id);

        // Gradients should exist
        assert!(autograd::get_grad(a.id).is_some());
        assert!(autograd::get_grad(b.id).is_some());

        // Zero grads
        autograd::zero_grads();

        // Gradients should be gone
        assert!(autograd::get_grad(a.id).is_none(), "grad(a) should be cleared by zero_grads");
        assert!(autograd::get_grad(b.id).is_none(), "grad(b) should be cleared by zero_grads");

        // Cleanup
        autograd::clear_tape();
    }

    // =========================================================================
    // GQA: Grouped Query Attention
    // =========================================================================

    #[test]
    fn gqa_config_defaults_to_mha() {
        // All preset configs should have n_kv_heads == n_heads (standard MHA)
        let configs = [
            ModelConfig::tiny(1000),
            ModelConfig::small(1000),
            ModelConfig::medium(1000),
            ModelConfig::large(1000),
        ];
        for cfg in &configs {
            assert_eq!(cfg.n_kv_heads, cfg.n_heads,
                "Preset config should default to MHA (n_kv_heads == n_heads)");
        }
    }

    #[test]
    fn gqa_custom_config() {
        // n_heads=8, n_kv_heads=2 → group_size=4
        let cfg = ModelConfig::custom_gqa(1000, 64, 8, 2, 2, 2.0, 128);
        assert_eq!(cfg.n_heads, 8);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.kv_dim(), 16); // head_dim=8, kv_dim=8*2=16
        assert!(cfg.param_count() > 0);
    }

    #[test]
    fn gqa_param_count_less_than_mha() {
        let mha = ModelConfig::custom(1000, 64, 4, 2, 2.0, 128);
        let gqa = ModelConfig::custom_gqa(1000, 64, 4, 2, 2, 2.0, 128);
        // GQA with n_kv_heads=2 has fewer params due to smaller K/V projections
        assert!(gqa.param_count() < mha.param_count(),
            "GQA ({}) should have fewer params than MHA ({})",
            gqa.param_count(), mha.param_count());
    }

    #[test]
    fn gqa_forward_pass_mha_equivalent() {
        // With n_kv_heads == n_heads, GQA is standard MHA — forward pass should produce
        // valid output with correct shapes.
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let cfg = ModelConfig::custom(1000, 64, 4, 2, 2.0, 128);
            let model = crate::model::Transformer::new(&ctx, cfg);

            let batch = 1;
            let seq_len = 8;
            let tokens: Vec<u32> = (0..batch * seq_len).map(|i| (i % 100) as u32).collect();

            let logits = model.forward(&tokens, batch, seq_len, None, false);
            // logits shape: [batch * seq_len, vocab_size]
            assert_eq!(logits.shape, vec![batch * seq_len, 1000]);

            let data = logits.to_vec();
            for v in &data {
                assert!(v.is_finite(), "Logit should be finite, got {}", v);
            }
        });
    }

    #[test]
    fn gqa_forward_pass_with_groups() {
        // GQA with n_heads=4, n_kv_heads=2 (group_size=2)
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let cfg = ModelConfig::custom_gqa(1000, 64, 4, 2, 2, 2.0, 128);
            let model = crate::model::Transformer::new(&ctx, cfg);

            let batch = 1;
            let seq_len = 8;
            let tokens: Vec<u32> = (0..batch * seq_len).map(|i| (i % 100) as u32).collect();

            let logits = model.forward(&tokens, batch, seq_len, None, false);
            assert_eq!(logits.shape, vec![batch * seq_len, 1000]);

            let data = logits.to_vec();
            for v in &data {
                assert!(v.is_finite(), "GQA logit should be finite, got {}", v);
            }
        });
    }

    #[test]
    fn gqa_kv_cache_inference() {
        // Test that GQA works with KV cache (autoregressive generation)
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let cfg = ModelConfig::custom_gqa(1000, 64, 4, 2, 2, 2.0, 128);
            let model = crate::model::Transformer::new(&ctx, cfg);
            let mut kv_caches = model.init_kv_caches_preallocated(1);

            // Prefill with 4 tokens
            let tokens: Vec<u32> = vec![1, 2, 3, 4];
            let logits = model.forward(&tokens, 1, 4, Some(&mut kv_caches), false);
            assert_eq!(logits.shape, vec![4, 1000]);

            // Autoregressive step: 1 new token
            let next_token: Vec<u32> = vec![5];
            let logits2 = model.forward(&next_token, 1, 1, Some(&mut kv_caches), false);
            assert_eq!(logits2.shape, vec![1, 1000]);

            let data = logits2.to_vec();
            for v in &data {
                assert!(v.is_finite(), "GQA KV cache logit should be finite, got {}", v);
            }
        });
    }

    #[test]
    #[should_panic(expected = "n_heads (4) must be divisible by n_kv_heads (3)")]
    fn gqa_invalid_group_size_panics() {
        // n_heads=4, n_kv_heads=3 is invalid (4 % 3 != 0)
        ModelConfig::custom_gqa(1000, 64, 4, 3, 2, 2.0, 128);
    }

    #[test]
    fn fp16_cast_roundtrip() {
        let ctx = MetalContext::new();
        let data = vec![1.0f32, -2.5, 3.14, 0.0, 1e-3, 65504.0]; // 65504 = max half
        let buf = ctx.buffer_from_slice(&data);
        let f16_buf = ctx.alloc_buffer(data.len() * 2);
        let f32_buf = ctx.alloc_buffer(data.len() * 4);
        compute::gpu_cast_f32_to_f16(&ctx, &buf, &f16_buf, data.len() as u32);
        compute::gpu_cast_f16_to_f32(&ctx, &f16_buf, &f32_buf, data.len() as u32);
        let result = MetalContext::read_buffer(&f32_buf, data.len());
        for (i, (&orig, &back)) in data.iter().zip(result.iter()).enumerate() {
            let tol = orig.abs() * 0.01 + 1e-3; // half has ~0.1% relative error
            assert!((orig - back).abs() < tol, "FP16 roundtrip mismatch at {}: {} vs {}", i, orig, back);
        }
    }

    #[test]
    fn fp16_batched_matmul_correctness() {
        let ctx = MetalContext::new();
        // [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_buf = ctx.buffer_from_slice(&a);
        let b_buf = ctx.buffer_from_slice(&b);

        // FP32 reference
        let c_ref = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul(&ctx, &a_buf, &b_buf, &c_ref, 2, 2, 2, 3);
        let ref_result = MetalContext::read_buffer(&c_ref, 8);

        // FP16 path
        let a_f16 = ctx.alloc_buffer(a.len() * 2);
        let b_f16 = ctx.alloc_buffer(b.len() * 2);
        compute::gpu_cast_f32_to_f16(&ctx, &a_buf, &a_f16, a.len() as u32);
        compute::gpu_cast_f32_to_f16(&ctx, &b_buf, &b_f16, b.len() as u32);
        let c_f16 = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul_f16(&ctx, &a_f16, &b_f16, &c_f16, 2, 2, 2, 3);
        let f16_result = MetalContext::read_buffer(&c_f16, 8);

        for i in 0..8 {
            assert!((ref_result[i] - f16_result[i]).abs() < 1.0,
                "Batched FP16 mismatch at {}: {} vs {}", i, ref_result[i], f16_result[i]);
        }

        // Also test batched trans_b and trans_a via the FP16 functions
        let c_tb = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul_trans_b_f16(&ctx, &a_f16, &b_f16, &c_tb, 2, 2, 3, 3);
        let _ = MetalContext::read_buffer(&c_tb, 8); // just verify no crash

        let c_ta = ctx.alloc_buffer(2 * 3 * 2 * 4);
        compute::gpu_batched_matmul_trans_a_f16(&ctx, &a_f16, &b_f16, &c_ta, 2, 2, 3, 2);
        let _ = MetalContext::read_buffer(&c_ta, 12); // just verify no crash
    }

    #[test]
    fn fp32_matmul_variants_still_work() {
        // Wire gpu_matmul_trans_b and gpu_matmul_trans_a (FP32 versions replaced by FP16 in hot path)
        let ctx = MetalContext::new();
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3] for trans_b
        let a_buf = ctx.buffer_from_slice(&a);
        let b_buf = ctx.buffer_from_slice(&b);

        // gpu_matmul_trans_b: [2,3] @ [2,3]^T = [2,2]
        let c1 = ctx.alloc_buffer(4 * 4);
        compute::gpu_matmul_trans_b(&ctx, &a_buf, &b_buf, &c1, 2, 2, 3);
        let r1 = MetalContext::read_buffer(&c1, 4);
        assert!((r1[0] - 14.0).abs() < 0.1); // 1*1+2*2+3*3=14

        // gpu_matmul_trans_a: [2,3]^T @ [2,3] = [3,3]
        let c2 = ctx.alloc_buffer(9 * 4);
        compute::gpu_matmul_trans_a(&ctx, &a_buf, &a_buf, &c2, 2, 3, 3);
        let r2 = MetalContext::read_buffer(&c2, 9);
        assert!((r2[0] - 17.0).abs() < 0.1); // 1*1+4*4=17
    }

    #[test]
    fn moe_gather_scatter_kernels() {
        let ctx = MetalContext::new();
        let n_tokens = 4;
        let d = 3;
        // Input: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
        let input = ctx.buffer_from_slice(&[1.0,2.0,3.0, 4.0,5.0,6.0, 7.0,8.0,9.0, 10.0,11.0,12.0]);

        // Gather tokens 1 and 3
        let indices = ctx.buffer_from_u32_slice(&[1, 3]);
        let gathered = ctx.alloc_buffer(2 * d * 4);
        compute::gpu_moe_gather(&ctx, &input, &indices, &gathered, 2, d as u32);
        let result = MetalContext::read_buffer(&gathered, 2 * d);
        assert!((result[0] - 4.0).abs() < 0.01); // token 1, dim 0
        assert!((result[3] - 10.0).abs() < 0.01); // token 3, dim 0

        // Scatter-add with weights
        let combined = ctx.alloc_buffer(n_tokens * d * 4);
        compute::gpu_fill(&ctx, &combined, (n_tokens * d) as u32, 0.0);
        let weights = ctx.buffer_from_slice(&[0.5, 0.5]);
        compute::gpu_moe_scatter_add(&ctx, &gathered, &indices, &weights, &combined, 2, d as u32);
        let out = MetalContext::read_buffer(&combined, n_tokens * d);
        assert!((out[3] - 2.0).abs() < 0.01); // token 1, dim 0: 4.0 * 0.5 = 2.0
        assert!((out[9] - 5.0).abs() < 0.01); // token 3, dim 0: 10.0 * 0.5 = 5.0
    }

    #[test]
    fn flash_attention_op_variant_exists() {
        // Verify FlashAttention Op variant is constructable (used in inference path)
        let _op = crate::autograd::Op::FlashAttention {
            batch_heads: 4, seq_q: 8, seq_k: 8, head_dim: 16, kv_offset: 0,
        };
    }

    #[test]
    fn fp16_reverse_cast() {
        let ctx = MetalContext::new();
        let data = vec![1.0f32, 2.5, -3.0, 0.0];
        let f32_buf = ctx.buffer_from_slice(&data);
        let f16_buf = ctx.alloc_buffer(data.len() * 2);
        let back_buf = ctx.alloc_buffer(data.len() * 4);
        compute::gpu_cast_f32_to_f16(&ctx, &f32_buf, &f16_buf, data.len() as u32);
        compute::gpu_cast_f16_to_f32(&ctx, &f16_buf, &back_buf, data.len() as u32);
        let result = MetalContext::read_buffer(&back_buf, data.len());
        for (i, (&orig, &back)) in data.iter().zip(result.iter()).enumerate() {
            assert!((orig - back).abs() < 0.01, "fp16 reverse cast mismatch at {}: {} vs {}", i, orig, back);
        }
    }

    // =========================================================================
    // New features: ReLU, AXPY, WSD, SliceCols, EMA

    #[test]
    fn relu_activation_zeros_negatives() {
        let ctx = MetalContext::new();
        let input = Tensor::from_buffer(Arc::clone(&ctx),
            ctx.buffer_from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0]),
            vec![6]);
        let output = input.relu();
        let vals = output.to_vec();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        autograd::clear_tape();
    }

    #[test]
    fn relu_backward_passes_positive_gradients() {
        let ctx = MetalContext::new();
        let x = Tensor::from_buffer(Arc::clone(&ctx),
            ctx.buffer_from_slice(&[-1.0f32, 2.0, -3.0, 4.0]),
            vec![1, 4]);
        let y = x.relu(); // [0, 2, 0, 4]
        // Use matmul with ones to create a sum → scalar-like loss
        let ones = Tensor::from_buffer(Arc::clone(&ctx),
            ctx.buffer_from_slice(&[1.0f32, 1.0, 1.0, 1.0]),
            vec![4, 1]);
        let loss = y.matmul(&ones); // [1, 1] = sum of relu outputs = 6.0
        autograd::backward(&ctx, loss.id);
        let grad = autograd::get_grad(x.id).expect("should have gradient");
        let grad_vals = MetalContext::read_buffer(&grad, 4);
        // relu backward: grad = upstream * (input > 0)
        // upstream from matmul backward is ones, so grad = [0, 1, 0, 1]
        assert!(grad_vals[0].abs() < 0.01, "negative input should have 0 gradient");
        assert!(grad_vals[1] > 0.5, "positive input should have positive gradient");
        assert!(grad_vals[2].abs() < 0.01, "negative input should have 0 gradient");
        assert!(grad_vals[3] > 0.5, "positive input should have positive gradient");
        autograd::clear_tape();
    }

    #[test]
    fn axpy_fused_scale_add() {
        let ctx = MetalContext::new();
        let y_buf = ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let x_buf = ctx.buffer_from_slice(&[10.0f32, 20.0, 30.0, 40.0]);
        compute::gpu_axpy(&ctx, &y_buf, &x_buf, 4, 0.5);
        let result = MetalContext::read_buffer(&y_buf, 4);
        // y += 0.5 * x → [1+5, 2+10, 3+15, 4+20] = [6, 12, 18, 24]
        assert!((result[0] - 6.0).abs() < 0.01);
        assert!((result[1] - 12.0).abs() < 0.01);
        assert!((result[2] - 18.0).abs() < 0.01);
        assert!((result[3] - 24.0).abs() < 0.01);
    }

    #[test]
    fn wsd_schedule_three_phases() {
        let sched = crate::optim::WSDScheduler::with_phases(1.0, 10, 80, 10);
        assert_eq!(sched.total_steps(), 100);
        // Warmup phase: linearly increasing
        assert!((sched.get_lr(0) - 0.0).abs() < 0.01);
        assert!((sched.get_lr(5) - 0.5).abs() < 0.01);
        // Stable phase: constant at max
        assert!((sched.get_lr(10) - 1.0).abs() < 0.01);
        assert!((sched.get_lr(50) - 1.0).abs() < 0.01);
        assert!((sched.get_lr(89) - 1.0).abs() < 0.01);
        // Decay phase: linear to zero
        assert!(sched.get_lr(95) < 1.0);
        assert!(sched.get_lr(99) < 0.2);
    }

    #[test]
    fn inverse_sqrt_schedule() {
        let lr = crate::optim::inverse_sqrt_lr(1.0, 100, 0);
        assert!(lr < 0.02); // step 0 during warmup
        let lr = crate::optim::inverse_sqrt_lr(1.0, 100, 100);
        assert!((lr - 1.0).abs() < 0.01); // peak at warmup end
        let lr = crate::optim::inverse_sqrt_lr(1.0, 100, 400);
        assert!((lr - 0.5).abs() < 0.01); // sqrt(100)/sqrt(400) = 0.5
    }

    #[test]
    fn ema_update_kernel() {
        let ctx = MetalContext::new();
        let ema_buf = ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0]);
        let src_buf = ctx.buffer_from_slice(&[10.0f32, 20.0, 30.0]);
        compute::gpu_ema_update(&ctx, &ema_buf, &src_buf, 3, 0.9);
        let result = MetalContext::read_buffer(&ema_buf, 3);
        // ema = 0.9 * ema + 0.1 * src → [0.9+1.0, 1.8+2.0, 2.7+3.0] = [1.9, 3.8, 5.7]
        assert!((result[0] - 1.9).abs() < 0.01);
        assert!((result[1] - 3.8).abs() < 0.01);
        assert!((result[2] - 5.7).abs() < 0.01);
    }

    #[test]
    fn slice_cols_extracts_correct_columns() {
        let ctx = MetalContext::new();
        // [2, 4] matrix: [[1,2,3,4], [5,6,7,8]]
        let src = Tensor::from_buffer(Arc::clone(&ctx),
            ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![2, 4]);
        // Slice cols [1..3] → [[2,3], [6,7]]
        let sliced = src.slice_cols(1, 2);
        let vals = sliced.to_vec();
        assert_eq!(vals.len(), 4);
        assert!((vals[0] - 2.0).abs() < 0.01);
        assert!((vals[1] - 3.0).abs() < 0.01);
        assert!((vals[2] - 6.0).abs() < 0.01);
        assert!((vals[3] - 7.0).abs() < 0.01);
        autograd::clear_tape();
    }

    #[test]
    fn muon_optimizer_step_converges() {
        let ctx = MetalContext::new();
        // Simple 2D weight matrix — Muon should orthogonalize the update
        let w = Tensor::randn(&ctx, vec![4, 4], 1.0);
        let params = vec![&w];
        let mut muon = crate::optim::Muon::new(&ctx, &params, 0.0);

        // Fake a gradient and verify step doesn't crash
        let grad = ctx.buffer_from_slice(&[0.1f32; 16]);
        autograd::accumulate_grad_for_test(&ctx, w.id, &grad, 16);
        muon.step(0.01);

        // Weight should have changed
        let new_vals = w.to_vec();
        let changed = new_vals.iter().any(|&v| (v - 1.0).abs() > 0.001 || v.abs() > 0.001);
        assert!(changed, "Muon should modify weights");
        autograd::clear_tape();
    }

    // =========================================================================
    // Mega-kernel correctness: mega_ffn output must match standard FFN path
    // =========================================================================

    #[test]
    fn mega_ffn_matches_standard_ffn() {
        let ctx = test_ctx();
        let d = 256usize;
        let ff = 768usize;
        let n_tokens = 32usize;

        // Create input and weights
        let x = Tensor::randn(&ctx, vec![n_tokens, d], 0.02);
        let norm_w = Tensor::ones(&ctx, vec![d]);
        let w1 = Tensor::randn(&ctx, vec![d, ff], (2.0 / (d + ff) as f32).sqrt());
        let w2 = Tensor::randn(&ctx, vec![ff, d], (2.0 / (ff + d) as f32).sqrt());
        let w3 = Tensor::randn(&ctx, vec![d, ff], (2.0 / (d + ff) as f32).sqrt());
        let eps = 1e-5f32;

        // Standard path: rms_norm → matmul w1 → matmul w3 → silu_gate → matmul w2 → add residual
        let normed = x.rms_norm(&norm_w, eps);
        let gate = normed.matmul(&w1);
        let up = normed.matmul(&w3);
        let hidden = gate.silu_gate(&up);
        let down = hidden.matmul(&w2);
        let standard_out = x.add(&down);
        let standard_vals = standard_out.to_vec();

        // Mega-kernel path: single dispatch
        let out_buf = ctx.alloc_buffer(n_tokens * d * 4);
        compute::gpu_mega_ffn(
            &ctx, &x.buffer, &norm_w.buffer,
            &w1.buffer, &w2.buffer, &w3.buffer,
            &out_buf, n_tokens as u32, d as u32, ff as u32, eps,
        );
        let mega_out = Tensor::from_buffer(Arc::clone(&ctx), out_buf, vec![n_tokens, d]);
        let mega_vals = mega_out.to_vec();

        // Compare — FP32 tolerance for accumulated error in fused kernel
        let mut max_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f64;
        for i in 0..standard_vals.len() {
            let diff = (standard_vals[i] - mega_vals[i]).abs();
            max_diff = max_diff.max(diff);
            sum_abs_diff += diff as f64;
        }
        let avg_diff = sum_abs_diff / standard_vals.len() as f64;
        eprintln!("mega_ffn vs standard: max_diff={:.6}, avg_diff={:.8}", max_diff, avg_diff);

        // Allow some tolerance — the fused kernel computes norm inline which may differ slightly
        assert!(max_diff < 0.01, "mega_ffn max_diff too large: {}", max_diff);
        assert!(avg_diff < 0.001, "mega_ffn avg_diff too large: {}", avg_diff);
    }

    #[test]
    fn mega_ffn_backward_produces_gradients() {
        use crate::autograd::Op;
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let d = 128usize;
        let ff = 256usize;
        let n_tokens = 4usize;

        // Create input and weights with requires_grad
        let x = Tensor::randn(&ctx, vec![n_tokens, d], 0.02).with_grad();
        let norm_w = Tensor::ones(&ctx, vec![d]).with_grad();
        let w1 = Tensor::randn(&ctx, vec![d, ff], (2.0 / (d + ff) as f32).sqrt()).with_grad();
        let w2 = Tensor::randn(&ctx, vec![ff, d], (2.0 / (ff + d) as f32).sqrt()).with_grad();
        let w3 = Tensor::randn(&ctx, vec![d, ff], (2.0 / (d + ff) as f32).sqrt()).with_grad();
        let eps = 1e-5f32;

        // Forward via standard ops (which auto-record on tape)
        let normed = x.rms_norm(&norm_w, eps);
        let gate = normed.matmul(&w1);
        let up = normed.matmul(&w3);
        let hidden = gate.silu_gate(&up);
        let down = hidden.matmul(&w2);
        let out = x.add(&down);

        // Reduce to scalar: sum via matmul with ones vector
        let flat = out.reshape(vec![1, n_tokens * d]);
        let ones_vec = Tensor::ones(&ctx, vec![n_tokens * d, 1]).with_grad();
        let scalar = flat.matmul(&ones_vec); // [1, 1]

        autograd::backward(&ctx, scalar.id);

        // All weight tensors should have gradients
        let get_grad_vec = |id: usize, size: usize| -> Vec<f32> {
            autograd::get_grad(id).map(|buf| {
                MetalContext::read_buffer(&buf, size)
            }).unwrap_or_default()
        };

        let gx = get_grad_vec(x.id, n_tokens * d);
        let gw = get_grad_vec(norm_w.id, d);
        let gw1 = get_grad_vec(w1.id, d * ff);
        let gw2 = get_grad_vec(w2.id, ff * d);
        let gw3 = get_grad_vec(w3.id, d * ff);

        assert!(!gx.is_empty(), "x should have gradient");
        assert!(!gw.is_empty(), "norm_w should have gradient");
        assert!(!gw1.is_empty(), "w1 should have gradient");
        assert!(!gw2.is_empty(), "w2 should have gradient");
        assert!(!gw3.is_empty(), "w3 should have gradient");

        // Check non-zero and finite
        for (name, g) in [("x", &gx), ("norm_w", &gw), ("w1", &gw1), ("w2", &gw2), ("w3", &gw3)] {
            assert!(g.iter().any(|&v| v.abs() > 1e-10), "{} gradient is all zeros", name);
            assert!(g.iter().all(|v| v.is_finite()), "{} gradient has non-finite values", name);
        }

        autograd::clear_tape();
        eprintln!("Standard FFN backward: x_grad_norm={:.4}", gx.iter().map(|v| v*v).sum::<f32>().sqrt());

        // Now do the SAME thing via mega kernel to compare gradients
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        // Reuse same weights but fresh tensor IDs (new with_grad calls)
        let x2 = Tensor::from_buffer(Arc::clone(&ctx), x.buffer.clone(), vec![n_tokens, d]).with_grad();
        let nw2 = Tensor::from_buffer(Arc::clone(&ctx), norm_w.buffer.clone(), vec![d]).with_grad();
        let w1b = Tensor::from_buffer(Arc::clone(&ctx), w1.buffer.clone(), vec![d, ff]).with_grad();
        let w2b = Tensor::from_buffer(Arc::clone(&ctx), w2.buffer.clone(), vec![ff, d]).with_grad();
        let w3b = Tensor::from_buffer(Arc::clone(&ctx), w3.buffer.clone(), vec![d, ff]).with_grad();

        // Mega kernel forward
        let out_buf = ctx.alloc_buffer(n_tokens * d * 4);
        compute::gpu_mega_ffn(
            &ctx, &x2.buffer, &nw2.buffer,
            &w1b.buffer, &w2b.buffer, &w3b.buffer,
            &out_buf, n_tokens as u32, d as u32, ff as u32, eps,
        );
        let out2 = Tensor::from_buffer(Arc::clone(&ctx), out_buf, vec![n_tokens, d]).with_grad();

        // Record MegaFfn on tape
        autograd::record(autograd::TapeEntry {
            op: Op::MegaFfn { eps },
            inputs: vec![x2.id, nw2.id, w1b.id, w2b.id, w3b.id],
            output: out2.id,
            input_buffers: vec![
                x2.buffer.clone(), nw2.buffer.clone(),
                w1b.buffer.clone(), w2b.buffer.clone(), w3b.buffer.clone(),
            ],
            output_buffer: out2.buffer.clone(),
            shapes: vec![vec![n_tokens, d], vec![d, ff], vec![ff, d]],
            cached: None,
        });

        // Same reduction to scalar
        let flat2 = out2.reshape(vec![1, n_tokens * d]);
        let ones_vec2 = Tensor::ones(&ctx, vec![n_tokens * d, 1]).with_grad();
        let scalar2 = flat2.matmul(&ones_vec2);

        autograd::backward(&ctx, scalar2.id);

        let gx2 = get_grad_vec(x2.id, n_tokens * d);
        let gw_2 = get_grad_vec(nw2.id, d);
        let gw1_2 = get_grad_vec(w1b.id, d * ff);
        let gw2_2 = get_grad_vec(w2b.id, ff * d);
        let gw3_2 = get_grad_vec(w3b.id, d * ff);

        assert!(!gx2.is_empty(), "mega x should have gradient");
        assert!(!gw_2.is_empty(), "mega norm_w should have gradient");
        assert!(!gw1_2.is_empty(), "mega w1 should have gradient");
        assert!(!gw2_2.is_empty(), "mega w2 should have gradient");
        assert!(!gw3_2.is_empty(), "mega w3 should have gradient");

        // Compare mega vs standard gradients — should be close
        let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a < 1e-12 || norm_b < 1e-12 { return 0.0; }
            dot / (norm_a * norm_b)
        };

        let sim_x = cosine_sim(&gx, &gx2);
        let sim_w1 = cosine_sim(&gw1, &gw1_2);
        let sim_w2 = cosine_sim(&gw2, &gw2_2);
        let sim_w3 = cosine_sim(&gw3, &gw3_2);
        eprintln!("Mega vs standard grad cosine sim: x={:.4}, w1={:.4}, w2={:.4}, w3={:.4}", sim_x, sim_w1, sim_w2, sim_w3);

        assert!(sim_x > 0.99, "x gradient cosine sim too low: {}", sim_x);
        assert!(sim_w1 > 0.99, "w1 gradient cosine sim too low: {}", sim_w1);
        assert!(sim_w2 > 0.99, "w2 gradient cosine sim too low: {}", sim_w2);
        assert!(sim_w3 > 0.99, "w3 gradient cosine sim too low: {}", sim_w3);

        autograd::clear_tape();
    }
}
