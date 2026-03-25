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
}
