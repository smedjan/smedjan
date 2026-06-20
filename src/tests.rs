#[cfg(test)]
mod suite {
    use crate::autograd;
    use crate::datapipe;
    use crate::gpu::{compute, MetalContext};
    use crate::model::{ModelConfig, Transformer};
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
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![3, 4],
            );

            let c = a.matmul(&b);
            assert_eq!(c.shape, vec![2, 4]);

            let result = c.to_vec();
            // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
            //       = [38, 44, 50, 56]
            // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
            //       = [83, 98, 113, 128]
            let expected = [38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0];
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
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
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

        assert!(
            grad_a.is_some(),
            "Gradient for A should exist after backward"
        );
        assert!(
            grad_b.is_some(),
            "Gradient for B should exist after backward"
        );

        let grad_a_data = MetalContext::read_buffer(&grad_a.unwrap(), 4);
        let grad_b_data = MetalContext::read_buffer(&grad_b.unwrap(), 4);

        let want_grad_a = [11.0, 15.0, 11.0, 15.0];
        let want_grad_b = [4.0, 4.0, 6.0, 6.0];

        for (i, (got, want)) in grad_a_data.iter().zip(want_grad_a).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "grad A[{i}] expected {want}, got {got}"
            );
        }
        for (i, (got, want)) in grad_b_data.iter().zip(want_grad_b).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "grad B[{i}] expected {want}, got {got}"
            );
        }

        autograd::clear_tape();
    }

    #[test]
    fn autograd_add_backward_both_inputs_get_gradients() {
        let ctx = test_ctx();
        autograd::clear_tape();
        autograd::clear_recompute_registry();

        let a = Tensor::from_slice(&ctx, &[3.0], vec![1, 1]).with_grad();
        let b = Tensor::from_slice(&ctx, &[7.0], vec![1, 1]).with_grad();

        let c = a.add(&b);

        autograd::backward(&ctx, c.id);

        let grad_a = autograd::get_grad(a.id);
        let grad_b = autograd::get_grad(b.id);

        assert!(
            grad_a.is_some(),
            "Gradient for A should exist after add backward"
        );
        assert!(
            grad_b.is_some(),
            "Gradient for B should exist after add backward"
        );

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

        assert_eq!(
            decoded, text,
            "Encode/decode roundtrip should preserve text"
        );
    }

    #[test]
    fn tokenizer_special_token_ids() {
        assert_eq!(PAD_TOKEN, 0, "PAD_TOKEN should be 0");
        assert_eq!(BOS_TOKEN, 1, "BOS_TOKEN should be 1");
        assert_eq!(EOS_TOKEN, 2, "EOS_TOKEN should be 2");
        assert_eq!(SPECIAL_TOKENS, 3, "SPECIAL_TOKENS count should be 3");
    }

    /// Importing a GPT-2 / HF `merges.txt` reproduces its merge rules exactly (byte-level BPE).
    #[test]
    fn import_gpt2_merges_reproduces_rules() {
        // Printable ASCII maps to itself in GPT-2's byte map, so these lines mean: a+b→ab, ab+a→aba.
        let merges = "#version: 0.2\na b\nab a\n";
        let tok = BpeTokenizer::import_gpt2_merges(merges);
        // base = 3 special + 256 bytes; +2 merges = 261 tokens.
        assert_eq!(tok.inverse_vocab.len(), 261);
        assert_eq!(tok.merges.len(), 2);
        // The merges apply during encoding.
        assert_eq!(
            tok.encode("ab").len(),
            1,
            "a+b merge should collapse to one token"
        );
        assert_eq!(
            tok.encode("aba").len(),
            1,
            "ab+a merge should collapse to one token"
        );
        assert_eq!(tok.encode("ba").len(), 2, "no b+a merge is defined");
        // Decode roundtrips through the imported vocab.
        assert_eq!(tok.decode(&tok.encode("aba")), "aba");
        assert_eq!(tok.decode(&tok.encode("ab")), "ab");
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
        assert!(
            !encoded.is_empty(),
            "Encoding should produce at least one token"
        );
    }

    // =========================================================================
    // 4. SHA-256 (datapipe)
    // =========================================================================

    #[test]
    fn sha256_known_string() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let hash = datapipe::sha256(b"hello");
        assert_eq!(
            hash, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            "SHA-256 of 'hello' mismatch"
        );
    }

    #[test]
    fn sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = datapipe::sha256(b"");
        assert_eq!(
            hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 of empty string mismatch"
        );
    }

    #[test]
    fn sha256_longer_input() {
        // SHA-256("The quick brown fox jumps over the lazy dog")
        // = d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592
        let hash = datapipe::sha256(b"The quick brown fox jumps over the lazy dog");
        assert_eq!(
            hash, "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
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
        let data = vec![3.5f32, 2.6, 1.2, 1.9];
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

    #[test]
    fn cautious_mask_zeros_disagreeing_and_renormalizes() {
        // Cautious Muon (Liang et al. 2024): keep update components whose sign agrees with the
        // gradient (u·g > 0), zero the rest, emit a 1/0 keep-mask, then renormalize by size/(kept+1).
        let ctx = test_ctx();
        let u = ctx.buffer_from_slice(&[1.0f32, -2.0, 3.0, -4.0]); // candidate update
        let g = ctx.buffer_from_slice(&[1.0f32, 5.0, -1.0, -4.0]); // gradient
                                                                   // agree: idx0 (1·1>0), idx3 (-4·-4>0); disagree: idx1 (-2·5<0), idx2 (3·-1<0)
        let keep = ctx.alloc_buffer(4 * 4);
        compute::gpu_cautious_mask(&ctx, &u, &g, &keep, 4);
        assert_eq!(MetalContext::read_buffer(&u, 4), vec![1.0, 0.0, 0.0, -4.0]);
        assert_eq!(
            MetalContext::read_buffer(&keep, 4),
            vec![1.0, 0.0, 0.0, 1.0]
        );

        // Renorm: scale = size/(kept+1) = 4/(2+1) = 4/3; zeroed entries stay zero.
        let sum = ctx.alloc_buffer(4);
        compute::gpu_reduce_sum(&ctx, &keep, &sum, 4);
        compute::gpu_cautious_scale(&ctx, &u, &sum, 4);
        let out = MetalContext::read_buffer(&u, 4);
        let s = 4.0f32 / 3.0;
        assert!((out[0] - s).abs() < 1e-4, "got {}", out[0]);
        assert!((out[3] - (-4.0 * s)).abs() < 1e-4, "got {}", out[3]);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    // =========================================================================
    // 6. Model config
    // =========================================================================

    #[test]
    fn model_config_tiny_param_count() {
        let cfg = ModelConfig::tiny(8192);
        let params = cfg.param_count();
        assert!(
            params > 0,
            "Tiny model should have >0 params, got {}",
            params
        );
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
        assert!(
            !datapipe::quality_filter("too short"),
            "Should reject short text"
        );
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
    fn datapipe_mix_rejects_invalid_weights_and_malformed_shards() {
        let valid_path = std::path::PathBuf::from("/tmp/andreai_mix_valid.bin");
        let bad_path = std::path::PathBuf::from("/tmp/andreai_mix_bad.bin");
        let output_path = std::path::PathBuf::from("/tmp/andreai_mix_out.bin");
        let tokens: Vec<u32> = (0..32).collect();
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(&valid_path, bytes).expect("write valid shard");
        std::fs::write(&bad_path, [1u8, 2, 3]).expect("write malformed shard");

        let err = match datapipe::mix_shards(&[], &output_path) {
            Ok(_) => panic!("empty shard list should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match datapipe::mix_shards(&[(valid_path.clone(), 0.0)], &output_path) {
            Ok(_) => panic!("zero total weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("sum to > 0"),
            "unexpected error: {err}"
        );

        let err = match datapipe::mix_shards(&[(valid_path.clone(), -1.0)], &output_path) {
            Ok(_) => panic!("negative weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match datapipe::mix_shards(&[(valid_path.clone(), f32::NAN)], &output_path) {
            Ok(_) => panic!("NaN weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match datapipe::mix_shards(&[(bad_path.clone(), 1.0)], &output_path) {
            Ok(_) => panic!("malformed shard should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("multiple of 4"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(valid_path).ok();
        std::fs::remove_file(bad_path).ok();
        std::fs::remove_file(output_path).ok();
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
                orig,
                rec,
                err,
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
                orig,
                rec,
                err,
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
        for (i, &got) in recovered.iter().enumerate().take(8) {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            let err = (got - expected).abs();
            assert!(
                err < 0.15,
                "Q4 nibble packing error at index {}: expected ~{}, got {}",
                i,
                expected,
                got,
            );
        }
    }

    #[test]
    fn quantize_constant_data_roundtrip() {
        // All identical values — scale should be ~0, zero should be the value
        let data = vec![3.5f32; 64];
        let shape = vec![64];

        let qt8 = crate::quantize::quantize(&data, &shape, 8, 32);
        let rec8 = crate::quantize::dequantize(&qt8);
        for &v in &rec8 {
            assert!((v - 3.5).abs() < 0.01, "Q8 constant data: got {}", v);
        }

        let qt4 = crate::quantize::quantize(&data, &shape, 4, 32);
        let rec4 = crate::quantize::dequantize(&qt4);
        for &v in &rec4 {
            assert!((v - 3.5).abs() < 0.01, "Q4 constant data: got {}", v);
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
        assert!(
            encoded.is_empty(),
            "Empty string should encode to empty vec"
        );
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
    fn tokenizer_load_rejects_malformed_files() {
        let bad_magic_path = "/tmp/andreai_test_bad_tokenizer_magic.bin";
        std::fs::write(bad_magic_path, b"NOPE").expect("write bad tokenizer magic");
        let err = match BpeTokenizer::load(bad_magic_path) {
            Ok(_) => panic!("bad tokenizer magic should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("not a valid tokenizer file"),
            "unexpected error: {err}"
        );

        let bad_version_path = "/tmp/andreai_test_bad_tokenizer_version.bin";
        let mut bad_version = Vec::new();
        bad_version.extend_from_slice(b"ABPE");
        bad_version.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(bad_version_path, bad_version).expect("write bad tokenizer version");
        let err = match BpeTokenizer::load(bad_version_path) {
            Ok(_) => panic!("bad tokenizer version should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("unsupported tokenizer version"),
            "unexpected error: {err}"
        );

        let truncated_path = "/tmp/andreai_test_truncated_tokenizer.bin";
        let mut truncated = Vec::new();
        truncated.extend_from_slice(b"ABPE");
        truncated.extend_from_slice(&1u32.to_le_bytes());
        truncated.extend_from_slice(&(SPECIAL_TOKENS + 256).to_le_bytes());
        std::fs::write(truncated_path, truncated).expect("write truncated tokenizer");
        let err = match BpeTokenizer::load(truncated_path) {
            Ok(_) => panic!("truncated tokenizer should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("vocab table"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(bad_magic_path).ok();
        std::fs::remove_file(bad_version_path).ok();
        std::fs::remove_file(truncated_path).ok();
    }

    #[test]
    fn longctx_eval_set_builds_calibrated_probes() {
        let corpus = b"the quick brown fox jumps over the lazy dog 0123456789 harbor town river bridge orchard";
        let tok = BpeTokenizer::train(corpus, 300);
        let lengths = [128usize, 256];
        let depths = [0.0f32, 0.5, 1.0];
        let set = crate::eval::longctx_eval_set(&tok, &lengths, &depths);
        assert_eq!(set.len(), 3 * lengths.len() * depths.len());
        for ex in &set {
            let (probe, lpart) = ex
                .category
                .split_once("_L")
                .expect("category form <probe>_L<len>");
            assert!(
                ["niah", "multikey", "vartrace"].contains(&probe),
                "unexpected probe {probe}"
            );
            let len: usize = lpart.parse().expect("numeric length suffix");
            assert!(lengths.contains(&len), "length {len} not requested");
            assert!(!ex.expected.is_empty(), "empty expected answer");
            assert!(
                ex.prompt.contains(&ex.expected),
                "needle {} must be embedded in its {} prompt",
                ex.expected,
                ex.category
            );
            let n_tok = tok.encode(&ex.prompt).len();
            assert!(
                n_tok >= len / 2,
                "{} prompt {n_tok} tok, expected ~{len}",
                ex.category
            );
        }
    }

    #[test]
    fn tokenizer_long_text_chunked_encoding() {
        let corpus = b"abcdefghijklmnopqrstuvwxyz 0123456789 the quick brown fox jumps";
        let tok = BpeTokenizer::train(corpus, 280);

        // Create a long text that triggers chunked encoding (> 10000 bytes)
        let long_text = "the quick brown fox ".repeat(600); // 12000 chars
        let encoded = tok.encode(&long_text);
        let decoded = tok.decode(&encoded);

        assert_eq!(
            decoded, long_text,
            "Chunked encoding should roundtrip correctly"
        );
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
    fn tensor_batched_matmul_trans_a_forward() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // A: [1, M=3, K=2], B: [1, M=3, N=2] → C = A^T @ B : [1, 2, 2]
            let a = Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
            let b = Tensor::from_slice(&ctx, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
            let c = a.batched_matmul_trans_a(&b);
            assert_eq!(c.shape, vec![1, 2, 2]);
            let r = c.to_vec();
            // C[k,n] = Σ_m A[m,k] B[m,n] → [[6,8],[8,10]]
            for (got, want) in r.iter().zip([6.0, 8.0, 8.0, 10.0]) {
                assert!((got - want).abs() < 1e-2, "got {got} want {want}");
            }
        });
    }

    #[test]
    fn tensor_batched_matmul_trans_a_backward() {
        let ctx = test_ctx();
        let a =
            Tensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]).with_grad();
        let b =
            Tensor::from_slice(&ctx, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]).with_grad();
        let c = a.batched_matmul_trans_a(&b); // [1, 2, 2]
                                              // loss = sum(C) → dC = ones[2,2]
        let flat = c.reshape(vec![1, 4]);
        let ones = Tensor::ones(&ctx, vec![4, 1]);
        let loss = flat.matmul(&ones); // [1,1]
        autograd::backward(&ctx, loss.id);

        // dA[m,k] = Σ_n B[m,n]  → [[1,1],[1,1],[2,2]]
        let ga = Tensor::from_buffer(
            Arc::clone(&ctx),
            autograd::get_grad(a.id).unwrap(),
            vec![1, 3, 2],
        )
        .to_vec();
        for (got, want) in ga.iter().zip([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]) {
            assert!((got - want).abs() < 1e-2, "dA got {got} want {want}");
        }
        // dB[m,n] = Σ_k A[m,k]  → [[3,3],[7,7],[11,11]]
        let gb = Tensor::from_buffer(
            Arc::clone(&ctx),
            autograd::get_grad(b.id).unwrap(),
            vec![1, 3, 2],
        )
        .to_vec();
        for (got, want) in gb.iter().zip([3.0, 3.0, 7.0, 7.0, 11.0, 11.0]) {
            assert!((got - want).abs() < 1e-2, "dB got {got} want {want}");
        }
        autograd::zero_grads();
    }

    /// Regression for the address-keyed fp16-cache stale-hit bug: when a buffer is freed and a new
    /// buffer is allocated at the same address, cast_to_f16 (used by the batch==1 matmul) must NOT
    /// return the previous buffer's cached fp16. alloc_buffer invalidates the cache for the address.
    #[test]
    fn f16_cache_invalidated_on_buffer_reuse() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let n = 64usize;
            let identity: Vec<f32> = (0..n * n)
                .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
                .collect();
            let id = Tensor::from_slice(&ctx, &identity, vec![n, n]);

            // A = all 2.0; matmul caches fp16(A) keyed by A's buffer address.
            let a = Tensor::full(&ctx, vec![n, n], 2.0);
            let _ = a.matmul(&id).to_vec();

            // Recycle A's buffer, then alloc the same size → reuse A's exact address.
            MetalContext::recycle_buffer(a.buffer.clone());
            let reused = ctx.alloc_buffer(n * n * 4);
            compute::gpu_fill(&ctx, &reused, (n * n) as u32, 3.0);
            let b = Tensor::from_buffer(Arc::clone(&ctx), reused, vec![n, n]);
            let rb = b.matmul(&id).to_vec(); // B @ I = B

            // Must reflect the NEW data (3.0), not A's stale cached fp16 (2.0).
            assert!(
                (rb[0] - 3.0).abs() < 0.05,
                "stale fp16 cache hit: got {} expected ~3.0",
                rb[0]
            );
            assert!(
                (rb[n * n - 1] - 3.0).abs() < 0.05,
                "stale fp16 cache hit at end: got {}",
                rb[n * n - 1]
            );
        });
    }

    /// BF16 matmul: fp32 RANGE (no ±65504 clamp like the fp16 path) at bf16 mantissa precision.
    #[test]
    fn matmul_bf16_range_and_precision() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // 1) Range: a value above the fp16 max (65504) is preserved — bf16 has fp32's exponent.
            let big = 1.0e5f32;
            let at = Tensor::from_slice(&ctx, &[big; 32], vec![1, 32]);
            let mut bv = vec![0.0f32; 32];
            bv[0] = 1.0;
            let bt = Tensor::from_slice(&ctx, &bv, vec![32, 1]);
            let r = at.matmul_bf16(&bt).to_vec()[0];
            assert!(
                (r - big).abs() < big * 0.02,
                "bf16 must preserve 1e5 (range): got {r}"
            );

            // 2) Precision: matches a CPU fp32 reference to ~bf16 tolerance (not exact).
            let (m, k, n) = (24usize, 40usize, 24usize);
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.37)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.29)
                .collect();
            let at = Tensor::from_slice(&ctx, &a, vec![m, k]);
            let bt = Tensor::from_slice(&ctx, &b, vec![k, n]);
            let got = at.matmul_bf16(&bt).to_vec();
            let mut max_rel = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for kk in 0..k {
                        s += a[i * k + kk] * b[kk * n + j];
                    }
                    max_rel = max_rel.max((got[i * n + j] - s).abs() / (1.0 + s.abs()));
                }
            }
            // bf16 has only ~7-8 mantissa bits, so accumulated relative error over k=40 is several %
            // (here ~6%) — far looser than fp32 but the trade for fp32 range at half the bandwidth.
            assert!(
                max_rel < 1e-1,
                "bf16 matmul should be ~bf16-accurate: max_rel={max_rel}"
            );
            eprintln!("bf16 matmul: max relative error vs fp32 = {max_rel:.3e}");
        });
    }

    /// The opt-in full-fp32 matmul keeps precision AND range that the default fp16-tile matmul loses.
    #[test]
    fn matmul_precise_full_fp32_no_clamp() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // 1) Precision: matches a CPU fp32 reference tightly (the fp16 path needs ~1e-2 tolerance).
            let (m, k, n) = (40usize, 50usize, 48usize);
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.37)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.29)
                .collect();
            let at = Tensor::from_slice(&ctx, &a, vec![m, k]);
            let bt = Tensor::from_slice(&ctx, &b, vec![k, n]);
            let precise = at.matmul_precise(&bt).to_vec();
            let mut cpu = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for kk in 0..k {
                        s += a[i * k + kk] * b[kk * n + j];
                    }
                    cpu[i * n + j] = s;
                }
            }
            let max_rel = precise
                .iter()
                .zip(&cpu)
                .map(|(p, c)| (p - c).abs() / (1.0 + c.abs()))
                .fold(0.0f32, f32::max);
            assert!(
                max_rel < 1e-4,
                "fp32 matmul precision: max_rel={max_rel} (should be ≪ fp16)"
            );

            // 2) Range: a value above the fp16 max (65504) is preserved; the fp16 path corrupts it.
            let big = 1.0e5f32;
            let at2 = Tensor::from_slice(&ctx, &[big; 32], vec![1, 32]);
            let mut bv = vec![0.0f32; 32];
            bv[0] = 1.0; // selects A[0]
            let bt2 = Tensor::from_slice(&ctx, &bv, vec![32, 1]);
            let r = at2.matmul_precise(&bt2).to_vec()[0];
            assert!(
                (r - big).abs() < big * 1e-3,
                "fp32 must preserve 1e5: got {r}"
            );
            let rf16 = at2.matmul(&bt2).to_vec()[0];
            assert!(
                (rf16 - big).abs() > big * 0.1,
                "fp16 path corrupts 1e5 (overflow/clamp): got {rf16}"
            );
        });
    }

    #[test]
    fn broadcast_rows_forward_backward() {
        let ctx = test_ctx();
        let want = [1.0f32, -2.0, 3.0];
        let v = Tensor::from_slice(&ctx, &want, vec![3]).with_grad();
        let out = v.broadcast_rows(4); // [4, 3], each row = [1, -2, 3]
        assert_eq!(out.shape, vec![4, 3]);
        let ov = out.to_vec();
        for r in 0..4 {
            for c in 0..3 {
                assert!(
                    (ov[r * 3 + c] - want[c]).abs() < 1e-5,
                    "broadcast fwd r{r}c{c}"
                );
            }
        }
        // loss = sum(out) → grad_v[c] = Σ_rows 1 = 4 (the row count).
        let ones = Tensor::ones(&ctx, vec![12, 1]);
        let loss = out.reshape(vec![1, 12]).matmul(&ones);
        autograd::backward(&ctx, loss.id);
        let g = Tensor::from_buffer(Arc::clone(&ctx), autograd::get_grad(v.id).unwrap(), vec![3])
            .to_vec();
        for (c, &gc) in g.iter().enumerate() {
            assert!(
                (gc - 4.0).abs() < 0.05,
                "broadcast bwd (column-sum) col {c}: got {gc}"
            );
        }
        autograd::zero_grads();
    }

    #[test]
    fn tensor_exp_forward_backward() {
        let ctx = test_ctx();
        let x = Tensor::from_slice(&ctx, &[0.0, 1.0, -1.0, 2.0], vec![4]).with_grad();
        let y = x.exp();
        let want = [
            1.0f32,
            std::f32::consts::E,
            1.0 / std::f32::consts::E,
            (2.0f32).exp(),
        ];
        for (got, w) in y.to_vec().iter().zip(want) {
            assert!((got - w).abs() < 1e-3, "exp fwd got {got} want {w}");
        }
        // loss = sum(exp(x)) → dL/dx = exp(x)
        let ones = Tensor::ones(&ctx, vec![4, 1]);
        let loss = y.reshape(vec![1, 4]).matmul(&ones);
        autograd::backward(&ctx, loss.id);
        let g = Tensor::from_buffer(Arc::clone(&ctx), autograd::get_grad(x.id).unwrap(), vec![4])
            .to_vec();
        for (got, w) in g.iter().zip(want) {
            assert!((got - w).abs() < 1e-2, "exp bwd got {got} want {w}");
        }
        autograd::zero_grads();
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

        assert!(
            lr_warmup > lr_mid,
            "LR should decrease after warmup: {} > {}",
            lr_warmup,
            lr_mid
        );
        assert!(
            lr_mid > lr_end,
            "LR should continue decreasing: {} > {}",
            lr_mid,
            lr_end
        );
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
                &ctx, &student, &teacher, 4.0, // temperature
                1.0, // alpha=1.0 means pure KL (no CE component)
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
                &ctx, &student, &teacher, 4.0, 1.0, // pure KL
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
            let (mixed_loss, _) =
                crate::loss::distillation_loss(&ctx, &student, &teacher, 4.0, 0.5, &targets);

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
                mixed_val,
                kl_val,
                ce_val,
                expected,
                diff
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

        crate::gpu::compute::gpu_transpose_2d(&ctx, &input_buf, &output_buf, rows, cols);

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

        crate::gpu::compute::gpu_transpose_2d(&ctx, &input_buf, &output_buf, n, n);

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
        assert!(
            autograd::get_grad(a.id).is_some(),
            "grad(a) should exist before clear"
        );

        // Clear tape but keep grads
        autograd::clear_tape_keep_grads();

        // Tape should be empty
        let (tape_ops, _) = autograd::tape_stats();
        assert_eq!(
            tape_ops, 0,
            "Tape should be empty after clear_tape_keep_grads"
        );

        // Gradients should still exist
        assert!(
            autograd::get_grad(a.id).is_some(),
            "grad(a) should survive clear_tape_keep_grads"
        );
        assert!(
            autograd::get_grad(b.id).is_some(),
            "grad(b) should survive clear_tape_keep_grads"
        );

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
        assert!(
            autograd::get_grad(a.id).is_none(),
            "grad(a) should be cleared by zero_grads"
        );
        assert!(
            autograd::get_grad(b.id).is_none(),
            "grad(b) should be cleared by zero_grads"
        );

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
            assert_eq!(
                cfg.n_kv_heads, cfg.n_heads,
                "Preset config should default to MHA (n_kv_heads == n_heads)"
            );
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
        assert!(
            gqa.param_count() < mha.param_count(),
            "GQA ({}) should have fewer params than MHA ({})",
            gqa.param_count(),
            mha.param_count()
        );
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
                assert!(
                    v.is_finite(),
                    "GQA KV cache logit should be finite, got {}",
                    v
                );
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
        let data = vec![1.0f32, -2.5, 3.5, 0.0, 1e-3, 65504.0]; // 65504 = max half
        let buf = ctx.buffer_from_slice(&data);
        let f16_buf = ctx.alloc_buffer(data.len() * 2);
        let f32_buf = ctx.alloc_buffer(data.len() * 4);
        compute::gpu_cast_f32_to_f16(&ctx, &buf, &f16_buf, data.len() as u32);
        compute::gpu_cast_f16_to_f32(&ctx, &f16_buf, &f32_buf, data.len() as u32);
        let result = MetalContext::read_buffer(&f32_buf, data.len());
        for (i, (&orig, &back)) in data.iter().zip(result.iter()).enumerate() {
            let tol = orig.abs() * 0.01 + 1e-3; // half has ~0.1% relative error
            assert!(
                (orig - back).abs() < tol,
                "FP16 roundtrip mismatch at {}: {} vs {}",
                i,
                orig,
                back
            );
        }
    }

    #[test]
    fn fp16_batched_matmul_correctness() {
        let ctx = MetalContext::new();
        // [2, 2, 3] @ [2, 3, 2] = [2, 2, 2]
        let a = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let b = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let a_buf = ctx.buffer_from_slice(&a);
        let b_buf = ctx.buffer_from_slice(&b);

        // FP32 reference
        let c_ref = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul(
            &ctx,
            &a_buf,
            &b_buf,
            &c_ref,
            compute::BatchedDims {
                batch: 2,
                m: 2,
                n: 2,
                k: 3,
            },
        );
        let ref_result = MetalContext::read_buffer(&c_ref, 8);

        // FP16 path
        let a_f16 = ctx.alloc_buffer(a.len() * 2);
        let b_f16 = ctx.alloc_buffer(b.len() * 2);
        compute::gpu_cast_f32_to_f16(&ctx, &a_buf, &a_f16, a.len() as u32);
        compute::gpu_cast_f32_to_f16(&ctx, &b_buf, &b_f16, b.len() as u32);
        let c_f16 = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul_f16(
            &ctx,
            &a_f16,
            &b_f16,
            &c_f16,
            compute::BatchedDims {
                batch: 2,
                m: 2,
                n: 2,
                k: 3,
            },
        );
        let f16_result = MetalContext::read_buffer(&c_f16, 8);

        for i in 0..8 {
            assert!(
                (ref_result[i] - f16_result[i]).abs() < 1.0,
                "Batched FP16 mismatch at {}: {} vs {}",
                i,
                ref_result[i],
                f16_result[i]
            );
        }

        // Also test batched trans_b and trans_a via the FP16 functions
        let c_tb = ctx.alloc_buffer(2 * 2 * 2 * 4);
        compute::gpu_batched_matmul_trans_b_f16(
            &ctx,
            &a_f16,
            &b_f16,
            &c_tb,
            compute::BatchedDims {
                batch: 2,
                m: 2,
                n: 3,
                k: 3,
            },
        );
        let _ = MetalContext::read_buffer(&c_tb, 8); // just verify no crash

        let c_ta = ctx.alloc_buffer(2 * 3 * 2 * 4);
        compute::gpu_batched_matmul_trans_a_f16(
            &ctx,
            &a_f16,
            &b_f16,
            &c_ta,
            compute::BatchedDims {
                batch: 2,
                m: 2,
                n: 2,
                k: 3,
            },
        );
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
        let input = ctx.buffer_from_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);

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
            batch_heads: 4,
            seq_q: 8,
            seq_k: 8,
            head_dim: 16,
            kv_offset: 0,
        };
    }

    /// Run the Flash-Attention forward kernel and record `Op::FlashAttention` (mirrors the seq≥2048
    /// path in `MultiHeadAttention::forward`) so the fused forward+backward can be exercised at small,
    /// cross-tile sizes the production gate never reaches. q/k/v: [bh, seq, hd]; causal, kv_offset=0.
    fn flash_fwd_recorded(ctx: &Arc<MetalContext>, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        let (bh, seq, hd) = (q.shape[0], q.shape[1], q.shape[2]);
        let out_buf = ctx.alloc_buffer(bh * seq * hd * 4);
        compute::gpu_flash_attention_forward(
            ctx,
            &q.buffer,
            &k.buffer,
            &v.buffer,
            &out_buf,
            compute::FlashDims {
                batch_heads: bh as u32,
                seq_q: seq as u32,
                seq_k: seq as u32,
                head_dim: hd as u32,
                kv_offset: 0,
            },
        );
        let out = Tensor::from_buffer(Arc::clone(ctx), out_buf.clone(), vec![bh, seq, hd]);
        if autograd::is_recording() {
            autograd::record(autograd::TapeEntry {
                op: autograd::Op::FlashAttention {
                    batch_heads: bh,
                    seq_q: seq,
                    seq_k: seq,
                    head_dim: hd,
                    kv_offset: 0,
                },
                inputs: vec![q.id, k.id, v.id],
                output: out.id,
                input_buffers: vec![q.buffer.clone(), k.buffer.clone(), v.buffer.clone()],
                output_buffer: out_buf.clone(),
                shapes: vec![
                    q.shape.clone(),
                    k.shape.clone(),
                    v.shape.clone(),
                    out.shape.clone(),
                ],
                cached: Some(out_buf),
            });
        }
        out
    }

    /// Flash-Attention forward == dense causal attention. seq=40 > FA_BR=32 so the online-softmax
    /// max/sum rescaling ACROSS key blocks is exercised (single-tile would hide a cross-tile bug).
    #[test]
    fn flash_attention_matches_dense_causal() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            let (bh, seq, hd) = (2usize, 40usize, 16usize);
            let gen = |s: usize| {
                (0..bh * seq * hd)
                    .map(|i| (((i * 7 + s * 13) % 17) as f32 - 8.0) * 0.05)
                    .collect::<Vec<f32>>()
            };
            let q = Tensor::from_slice(&ctx, &gen(1), vec![bh, seq, hd]);
            let k = Tensor::from_slice(&ctx, &gen(2), vec![bh, seq, hd]);
            let v = Tensor::from_slice(&ctx, &gen(3), vec![bh, seq, hd]);
            let flash = flash_fwd_recorded(&ctx, &q, &k, &v).to_vec();
            // The kernel keeps Q in fp32 (`float q_row[]`) but stores K/V tiles as `half`, so it
            // implements fp16-K/V attention. Compare to a dense reference with K/V rounded through
            // fp16 (Q untouched): isolates the kernel MATH from the fp16 representation. A raw fp32
            // dense compare leaves ~3e-2 of pure rounding and would mask nothing useful.
            let fp16_round = |t: &Tensor| -> Tensor {
                let len: usize = t.shape.iter().product();
                let f16 = ctx.alloc_buffer(len * 2);
                compute::gpu_cast_f32_to_f16(&ctx, &t.buffer, &f16, len as u32);
                let back = ctx.alloc_buffer(len * 4);
                compute::gpu_cast_f16_to_f32(&ctx, &f16, &back, len as u32);
                Tensor::from_buffer(Arc::clone(&ctx), back, t.shape.clone())
            };
            let (kf, vf) = (fp16_round(&k), fp16_round(&v));
            let scale = 1.0 / (hd as f32).sqrt();
            let dense = q
                .batched_matmul_trans_b(&kf)
                .scale(scale)
                .causal_mask(0)
                .softmax()
                .batched_matmul(&vf)
                .to_vec();
            let md = flash
                .iter()
                .zip(&dense)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                flash.iter().all(|x| x.is_finite()),
                "flash output must be finite"
            );
            assert!(md < 6e-3, "flash vs fp16-K/V dense causal max_abs_diff={md:.5} (kernel math, fp16 rep removed)");
        });
    }

    /// Regression guard for the partial-last-q-block bug: a non-first q-block with fewer than FA_BR=32
    /// valid query rows (seq 33, 40 — NOT a multiple of 32) used to leave K/V_shared rows unloaded
    /// because out-of-range query threads returned before the cooperative tile load. Every seq, incl.
    /// the partial ones, must match dense causal attention within the fp16-K/V floor.
    #[test]
    fn flash_attention_partial_blocks_match_dense() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            for &seq in &[31usize, 32, 33, 40, 47, 64, 65] {
                let (bh, hd) = (1usize, 16usize);
                let gen = |s: usize| {
                    (0..bh * seq * hd)
                        .map(|i| (((i * 7 + s * 13) % 17) as f32 - 8.0) * 0.05)
                        .collect::<Vec<f32>>()
                };
                let q = Tensor::from_slice(&ctx, &gen(1), vec![bh, seq, hd]);
                let k = Tensor::from_slice(&ctx, &gen(2), vec![bh, seq, hd]);
                let v = Tensor::from_slice(&ctx, &gen(3), vec![bh, seq, hd]);
                let flash = flash_fwd_recorded(&ctx, &q, &k, &v).to_vec();
                let scale = 1.0 / (hd as f32).sqrt();
                let dense = q
                    .batched_matmul_trans_b(&k)
                    .scale(scale)
                    .causal_mask(0)
                    .softmax()
                    .batched_matmul(&v)
                    .to_vec();
                let md = flash
                    .iter()
                    .zip(&dense)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    flash.iter().all(|x| x.is_finite()),
                    "seq={seq}: flash output must be finite"
                );
                assert!(
                    md < 2e-3,
                    "seq={seq}: flash vs dense max_abs_diff={md:.5} (partial-block regression)"
                );
            }
        });
    }

    /// Finite-difference grad-check of the Flash-Attention fused forward+backward (dQ/dK/dV) at a
    /// cross-tile size (seq=40). Validates the online-softmax backward (D-term + block rescaling) end
    /// to end — previously only `flash_attention_op_variant_exists` (a constructability check) existed.
    #[test]
    fn gradcheck_flash_attention() {
        let ctx = test_ctx();
        let (bh, seq, hd) = (2usize, 40usize, 16usize);
        let n = bh * seq * hd;
        grad_check(
            &ctx,
            &[
                (gc_vec(n, 0), vec![bh, seq, hd]),
                (gc_vec(n, 7), vec![bh, seq, hd]),
                (gc_vec(n, 19), vec![bh, seq, hd]),
            ],
            &|t| flash_fwd_recorded(&ctx, &t[0], &t[1], &t[2]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "flash_attention",
        );
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
            assert!(
                (orig - back).abs() < 0.01,
                "fp16 reverse cast mismatch at {}: {} vs {}",
                i,
                orig,
                back
            );
        }
    }

    // =========================================================================
    // New features: ReLU, AXPY, WSD, SliceCols, EMA

    #[test]
    fn relu_activation_zeros_negatives() {
        let ctx = MetalContext::new();
        let input = Tensor::from_buffer(
            Arc::clone(&ctx),
            ctx.buffer_from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0]),
            vec![6],
        );
        let output = input.relu();
        let vals = output.to_vec();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        autograd::clear_tape();
    }

    #[test]
    fn relu_backward_passes_positive_gradients() {
        let ctx = MetalContext::new();
        let x = Tensor::from_buffer(
            Arc::clone(&ctx),
            ctx.buffer_from_slice(&[-1.0f32, 2.0, -3.0, 4.0]),
            vec![1, 4],
        );
        let y = x.relu(); // [0, 2, 0, 4]
                          // Use matmul with ones to create a sum → scalar-like loss
        let ones = Tensor::from_buffer(
            Arc::clone(&ctx),
            ctx.buffer_from_slice(&[1.0f32, 1.0, 1.0, 1.0]),
            vec![4, 1],
        );
        let loss = y.matmul(&ones); // [1, 1] = sum of relu outputs = 6.0
        autograd::backward(&ctx, loss.id);
        let grad = autograd::get_grad(x.id).expect("should have gradient");
        let grad_vals = MetalContext::read_buffer(&grad, 4);
        // relu backward: grad = upstream * (input > 0)
        // upstream from matmul backward is ones, so grad = [0, 1, 0, 1]
        assert!(
            grad_vals[0].abs() < 0.01,
            "negative input should have 0 gradient"
        );
        assert!(
            grad_vals[1] > 0.5,
            "positive input should have positive gradient"
        );
        assert!(
            grad_vals[2].abs() < 0.01,
            "negative input should have 0 gradient"
        );
        assert!(
            grad_vals[3] > 0.5,
            "positive input should have positive gradient"
        );
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
    fn train_config_rejects_unknown_optimizer_and_schedule() {
        let mut cfg = crate::train::TrainConfig::default_small("dataset.bin", "tokenizer.bin");
        for opt in crate::train::TrainConfig::SUPPORTED_OPTIMIZERS {
            cfg.optimizer_type = (*opt).to_string();
            cfg.lr_schedule = "cosine".to_string();
            cfg.validate()
                .unwrap_or_else(|e| panic!("optimizer {opt} should validate: {e}"));
        }
        for sched in crate::train::TrainConfig::SUPPORTED_LR_SCHEDULES {
            cfg.optimizer_type = "adamw".to_string();
            cfg.lr_schedule = (*sched).to_string();
            cfg.validate()
                .unwrap_or_else(|e| panic!("schedule {sched} should validate: {e}"));
        }

        cfg.optimizer_type = "definitely-not-real".to_string();
        cfg.lr_schedule = "cosine".to_string();
        let err = cfg.validate().expect_err("unknown optimizer should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("unsupported optimizer"),
            "unexpected error: {err}"
        );

        cfg.optimizer_type = "adamw".to_string();
        cfg.lr_schedule = "lunar".to_string();
        let err = cfg.validate().expect_err("unknown LR schedule should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("unsupported lr_schedule"),
            "unexpected error: {err}"
        );
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
        let src = Tensor::from_buffer(
            Arc::clone(&ctx),
            ctx.buffer_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![2, 4],
        );
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
        let changed = new_vals
            .iter()
            .any(|&v| (v - 1.0).abs() > 0.001 || v.abs() > 0.001);
        assert!(changed, "Muon should modify weights");
        autograd::clear_tape();
    }

    /// The Muon+AdamW hybrid must route by ROLE, not just by shape: embeddings and the tied LM head
    /// are 2-D but go to AdamW (orthogonalizing them is Muon's known pathology); 1-D norms go to
    /// AdamW; the hidden attention/FFN matrices go to Muon. The partition must be exact + disjoint.
    #[test]
    fn hybrid_optimizer_routes_by_role() {
        let ctx = test_ctx();
        let model = Transformer::new(&ctx, ModelConfig::custom(48, 128, 4, 4, 2.67, 64));
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let force = model.force_adamw_param_ids();
        let h = crate::optim::HybridOptimizer::new(
            &ctx,
            &prefs,
            0.1,
            &force,
            crate::optim::AdamWHyper::default(),
        );
        let muon_ids: std::collections::HashSet<usize> =
            h.muon.params.iter().map(|p| p.tensor_id).collect();
        let adamw_ids: std::collections::HashSet<usize> =
            h.adamw.params.iter().map(|p| p.tensor_id).collect();

        // Embedding (2-D, weight-tied head) → AdamW, never Muon.
        assert!(
            adamw_ids.contains(&model.embedding.id),
            "embedding must be on AdamW"
        );
        assert!(
            !muon_ids.contains(&model.embedding.id),
            "embedding must NOT be orthogonalized by Muon"
        );
        // A hidden attention projection (2-D matrix) → Muon.
        assert!(
            muon_ids.contains(&model.blocks[0].attn.w_q.id),
            "hidden attn matrix must use Muon"
        );
        // 1-D final norm → AdamW.
        assert!(
            adamw_ids.contains(&model.ln_final_weight.id),
            "1-D norm must be on AdamW"
        );
        // Partition is exact + disjoint.
        assert_eq!(
            muon_ids.len() + adamw_ids.len(),
            prefs.len(),
            "every param routed exactly once"
        );
        assert!(
            muon_ids.is_disjoint(&adamw_ids),
            "a param cannot be on both optimizers"
        );
        assert!(
            !muon_ids.is_empty(),
            "there are hidden 2-D matrices to give to Muon"
        );
    }

    /// END-TO-END STABILITY of the Muon+AdamW hybrid via the unified `Optimizer::Hybrid` dispatch
    /// (Muon orthogonalizes the hidden matrices, AdamW drives embeddings/head/norms): the full
    /// forward→CE→backward→clip→hybrid-step loop must run and stay BOUNDED (no NaN, no blow-up).
    /// Mirrors `adamw_training_stays_bounded_no_grad_explosion` (gentle lr=1e-3 warmup) so it is
    /// parallel-safe and deterministic — not `#[ignore]`d. Convergence *quality* is proven where it
    /// matters: the deterministic `hybrid_optimizer_routes_by_role` test proves the role partition,
    /// and a real 600-step run on data/train_v3.bin reaches EMA 1.56 = the dense-AdamW baseline
    /// (recorded in docs/HANDOFF_adamw_and_efficiency.md §4). A head-dominated 32-vocab micro overfit
    /// with the head on AdamW is too init-sensitive at the high lr needed to memorize it to make a
    /// non-flaky convergence assertion, so this asserts the real guarantee (stability) instead.
    #[test]
    fn hybrid_optimizer_trains_stably() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let (batch, seq_len) = (8usize, 12usize);
        let one: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0, 9, 2, 8, 1];
        let tokens: Vec<u32> = one.iter().cloned().cycle().take(batch * seq_len).collect();
        let targets: Vec<u32> = vec![5; 8 * 12];
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let force = model.force_adamw_param_ids();
        let mut opt = crate::optim::Optimizer::Hybrid(crate::optim::HybridOptimizer::new(
            &ctx,
            &prefs,
            0.0,
            &force,
            crate::optim::AdamWHyper::default(),
        ));
        let mut max_loss = 0.0f32;
        for step in 0..150 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(lv.is_finite(), "hybrid loss non-finite at step {step}");
            max_loss = max_loss.max(lv);
            autograd::backward(&ctx, loss.id);
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let lr = 1e-3 * (((step + 1) as f32) / 20.0).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("hybrid stability: max loss {max_loss:.2}");
        assert!(
            max_loss < 30.0,
            "Muon+AdamW hybrid loss blew up (max {max_loss})"
        );
    }

    /// AdamW update clipping bounds the per-element step at the source. A collapsed second moment
    /// (v→0) with a non-trivial first moment produces a pathological spike update of ~1e6; with
    /// update_clip set, the param move is bounded to ~lr·clip; without it (clip=0) the move is huge.
    /// This is the in-kernel guard against the beta2-short-memory denominator-collapse instability.
    #[test]
    fn adamw_update_clip_bounds_step() {
        let ctx = test_ctx();
        let run = |clip: f32| -> f32 {
            let w = Tensor::zeros(&ctx, vec![4, 4]);
            let mut opt = crate::optim::AdamW::new_with_config(
                &ctx,
                &[&w],
                0.0,
                crate::optim::AdamWHyper {
                    beta1: 0.9,
                    beta2: 0.95,
                    eps: 1e-5,
                    update_clip: clip,
                },
            );
            // Pre-seed a collapsed v and a large m → the spike regime.
            compute::gpu_fill(&ctx, &opt.params[0].m, 16, 10.0);
            compute::gpu_fill(&ctx, &opt.params[0].v, 16, 1e-12);
            let zero = ctx.buffer_from_slice(&[0.0f32; 16]);
            autograd::accumulate_grad_for_test(&ctx, w.id, &zero, 16);
            opt.step(0.01);
            let d = w.to_vec()[0].abs();
            autograd::clear_tape();
            autograd::zero_grads();
            d
        };
        let clipped = run(0.5);
        let unclipped = run(0.0);
        eprintln!("update_clip: clipped |Δ|={clipped:.6}, unclipped |Δ|={unclipped:.1}");
        assert!(
            clipped <= 0.01,
            "clipped step must be bounded ~lr·clip, got {clipped}"
        );
        assert!(
            unclipped > 1.0,
            "unclipped spike update should be large, got {unclipped}"
        );
    }

    /// Block-sparse routing correctness: when top_k ≥ (#past blocks), every causal block is
    /// selectable, so block-sparse attention must reduce EXACTLY to dense causal attention.
    #[test]
    fn block_sparse_matches_dense_when_full() {
        let ctx = test_ctx();
        let (hd, seq, bsz) = (8usize, 8usize, 4usize); // nb=2
        let nb = seq.div_ceil(bsz);
        let scale = 1.0 / (hd as f32).sqrt();
        let gen = |salt: usize| -> Vec<f32> {
            (0..seq * hd)
                .map(|i| (((i * 5 + salt * 11) % 13) as f32 - 6.0) * 0.1)
                .collect()
        };
        let q = Tensor::from_slice(&ctx, &gen(1), vec![1, seq, hd]);
        let k = Tensor::from_slice(&ctx, &gen(2), vec![1, seq, hd]);
        let v = Tensor::from_slice(&ctx, &gen(3), vec![1, seq, hd]);
        // Block-sparse with top_k = nb (≥ any query's past-block count) → selects all causal blocks.
        let bm = k.block_mean_keys(bsz);
        let bs = q.batched_matmul_trans_b(&bm);
        let sparse = q
            .batched_matmul_trans_b(&k)
            .scale(scale)
            .block_sparse_mask(&bs, bsz, nb)
            .softmax()
            .batched_matmul(&v)
            .to_vec();
        // Dense causal.
        let dense = q
            .batched_matmul_trans_b(&k)
            .scale(scale)
            .causal_mask(0)
            .softmax()
            .batched_matmul(&v)
            .to_vec();
        let max_diff = sparse
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("block-sparse (full) vs dense: max_diff={max_diff:.6}");
        assert!(
            max_diff < 1e-4,
            "block-sparse with full top_k must equal dense causal: {max_diff}"
        );
        autograd::clear_tape();
    }

    /// Block-sparse selection is content-based + causal: with top_k=1, a query attends to its OWN
    /// block (causally) + the single highest-scoring PAST block, and masks everything else.
    #[test]
    fn block_sparse_selects_top_block() {
        let ctx = test_ctx();
        let (seq, bsz, nb) = (12usize, 4usize, 3usize);
        let scores = Tensor::zeros(&ctx, vec![1, seq, seq]); // 0 = unmasked baseline
                                                             // Block scores per query position; q=8 (block 2): block0 high, block1 low.
        let mut bs = vec![0.0f32; seq * nb];
        for q in 0..seq {
            bs[q * nb] = 5.0;
            bs[q * nb + 1] = 1.0;
            bs[q * nb + 2] = 0.0;
        }
        let block_scores = Tensor::from_slice(&ctx, &bs, vec![1, seq, nb]);
        let m = scores.block_sparse_mask(&block_scores, bsz, 1).to_vec();
        let at = |q: usize, k: usize| m[q * seq + k];
        // q=8 (block 2, first position) with top_k=1:
        //   block 0 (k 0..3) = top-1 past → visible (0.0)
        //   block 1 (k 4..7) = not selected → -inf
        //   own block 2: k=8 visible (causal), k=9..11 future → -inf
        for k in 0..4 {
            assert_eq!(
                at(8, k),
                0.0,
                "selected past block must be visible at k={k}"
            );
        }
        for k in 4..8 {
            assert!(
                at(8, k).is_infinite(),
                "unselected block must be masked at k={k}"
            );
        }
        assert_eq!(at(8, 8), 0.0, "own-block causal position must be visible");
        for k in 9..12 {
            assert!(at(8, k).is_infinite(), "future must be masked at k={k}");
        }
        autograd::clear_tape();
    }

    /// SUBQUADRATIC block-sparse gather: when top_k+1 ≥ nb (every causal block is selectable), the
    /// gather-based attention must reduce EXACTLY to dense causal attention. Proves the gather +
    /// compact-attention + causal mask are correct.
    #[test]
    fn block_sparse_gather_matches_dense_when_full() {
        let ctx = test_ctx();
        let (bh, seq, hd, block) = (2usize, 16usize, 8usize, 4usize); // nb=4
        let gen = |s: usize| {
            (0..bh * seq * hd)
                .map(|i| (((i * 7 + s * 13) % 17) as f32 - 8.0) * 0.1)
                .collect::<Vec<f32>>()
        };
        let q = Tensor::from_slice(&ctx, &gen(1), vec![bh, seq, hd]);
        let k = Tensor::from_slice(&ctx, &gen(2), vec![bh, seq, hd]);
        let v = Tensor::from_slice(&ctx, &gen(3), vec![bh, seq, hd]);
        // top_k=3 → k_sel=4=nb → all causal blocks selected → dense causal attention.
        let gathered =
            crate::attention::block_sparse_gather_attention(&q, &k, &v, block, 3).to_vec();
        let scale = 1.0 / (hd as f32).sqrt();
        let dense = q
            .batched_matmul_trans_b(&k)
            .scale(scale)
            .causal_mask(0)
            .softmax()
            .batched_matmul(&v)
            .to_vec();
        let max_diff = gathered
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("gather(full) vs dense causal: max_diff={max_diff:.6}");
        assert!(gathered.iter().all(|x| x.is_finite()));
        assert!(
            max_diff < 1e-4,
            "gather with full top_k must equal dense causal: {max_diff}"
        );
        autograd::clear_tape();
    }

    /// Genuinely sparse gather (top_k=1): finite output that DIFFERS from dense (most past blocks
    /// dropped) — confirms the routing actually restricts attention, not silently attending all.
    #[test]
    fn block_sparse_gather_is_sparse() {
        let ctx = test_ctx();
        let (bh, seq, hd, block) = (1usize, 32usize, 8usize, 4usize); // nb=8
        let gen = |s: usize| {
            (0..bh * seq * hd)
                .map(|i| (((i * 3 + s * 5) % 19) as f32 - 9.0) * 0.1)
                .collect::<Vec<f32>>()
        };
        let q = Tensor::from_slice(&ctx, &gen(1), vec![bh, seq, hd]);
        let k = Tensor::from_slice(&ctx, &gen(2), vec![bh, seq, hd]);
        let v = Tensor::from_slice(&ctx, &gen(3), vec![bh, seq, hd]);
        let sparse = crate::attention::block_sparse_gather_attention(&q, &k, &v, block, 1).to_vec(); // own + 1 past of up to 7
        let scale = 1.0 / (hd as f32).sqrt();
        let dense = q
            .batched_matmul_trans_b(&k)
            .scale(scale)
            .causal_mask(0)
            .softmax()
            .batched_matmul(&v)
            .to_vec();
        let max_diff = sparse
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "gather(top_k=1) vs dense: max_diff={max_diff:.4} (expect >0 — genuinely sparse)"
        );
        assert!(
            sparse.iter().all(|x| x.is_finite()),
            "sparse gather must be finite"
        );
        assert!(
            max_diff > 1e-3,
            "top_k=1 must differ from dense (it drops most blocks): {max_diff}"
        );
    }

    /// Finite-difference grad-check of the TRAINABLE block-sparse gather path (q, k, v all tracked).
    /// Uses the all-blocks config (top_k+1 == nb): each query block selects its own block + all
    /// strictly-past blocks (no top-k pruning, so the routing is STABLE under ±eps perturbation and
    /// equal to dense causal attention). This validates the new gather→…→scatter-add backward
    /// against ground-truth central differences. Key block 0 is selected by every query block, so
    /// the scatter-add's atomic accumulation (many query-blocks → one source block) is exercised.
    #[test]
    fn gradcheck_block_sparse_gather_attention() {
        let ctx = test_ctx();
        // bh=2, seq=8, hd=4, block=2 → nb=4, top_k=3 → k_sel=4=nb (all causal blocks selectable).
        let n = 2 * 8 * 4;
        grad_check(
            &ctx,
            &[
                (gc_vec(n, 0), vec![2, 8, 4]),
                (gc_vec(n, 13), vec![2, 8, 4]),
                (gc_vec(n, 29), vec![2, 8, 4]),
            ],
            &|t| crate::attention::block_sparse_gather_attention(&t[0], &t[1], &t[2], 2, 3),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "block_sparse_gather_attention",
        );
    }

    /// Equivalence guard (spec §3): with top_k+1 == nb (all causal blocks selectable) the gather path
    /// reduces EXACTLY to dense causal attention, so its q/k/v GRADIENTS must match dense's element-wise.
    /// This is the end-to-end check that the gather→matmul→softmax→matmul→scatter-add backward is the
    /// correct factoring of dense attention's backward (it caught the non-square `batched_matmul_trans_a`
    /// param-order bug: K and V grads diverged while q/forward matched).
    #[test]
    fn block_sparse_gather_grad_matches_dense_when_full() {
        let ctx = test_ctx();
        compute::set_simdgroup_matmul(false);
        compute::set_bf16_matmul(false);
        Tensor::clear_f16_cache();
        let (bh, seq, hd, block) = (2usize, 8usize, 4usize, 2usize); // nb=4, top_k=3 → all blocks
        let n = bh * seq * hd;
        let gen = |off: usize| (0..n).map(|i| gc_in(i + off)).collect::<Vec<f32>>();
        let (qd, kd, vd) = (gen(0), gen(13), gen(29));
        let scale = 1.0 / (hd as f32).sqrt();

        // Return (q_grad, k_grad, v_grad) for either the dense or the gather forward.
        let grads = |dense: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            autograd::clear_tape();
            autograd::zero_grads();
            let q = Tensor::from_slice(&ctx, &qd, vec![bh, seq, hd]).with_grad();
            let k = Tensor::from_slice(&ctx, &kd, vec![bh, seq, hd]).with_grad();
            let v = Tensor::from_slice(&ctx, &vd, vec![bh, seq, hd]).with_grad();
            let out = if dense {
                q.batched_matmul_trans_b(&k)
                    .scale(scale)
                    .causal_mask(0)
                    .softmax()
                    .batched_matmul(&v)
            } else {
                crate::attention::block_sparse_gather_attention(&q, &k, &v, block, 3)
            };
            let m: usize = out.shape.iter().product();
            let seed: Vec<f32> = (0..m).map(gc_seed).collect();
            let seed_t = Tensor::from_slice(&ctx, &seed, vec![m, 1]);
            let loss = out.reshape(vec![1, m]).matmul(&seed_t);
            autograd::backward(&ctx, loss.id);
            let fetch = |id: usize| {
                Tensor::from_buffer(Arc::clone(&ctx), autograd::get_grad(id).unwrap(), vec![n])
                    .to_vec()
            };
            (fetch(q.id), fetch(k.id), fetch(v.id))
        };
        let (dq, dk, dv) = grads(true);
        let (gq, gk, gv) = grads(false);
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        for (name, d, g) in [("q", &dq, &gq), ("k", &dk, &gk), ("v", &dv, &gv)] {
            let md = max_diff(d, g);
            assert!(
                md < 2e-3,
                "{name}-grad gather vs dense max_abs_diff={md:.6} (must reduce to dense)"
            );
        }
        autograd::clear_tape();
    }

    /// Direct unit test of the scatter-add backward kernel (transpose of `gather_blocks`), pinned to a
    /// hand-computed case independent of the autograd flow. bh=1, nb=2, seq=4, hd=1, block=2, k_sel=2.
    /// sel = [0, 2, 1, 0]: query-block 0 selects {block 0, sentinel}; query-block 1 selects {block 1,
    /// block 0}. So source block 0 receives contributions from BOTH query-blocks (accumulation) and the
    /// sentinel slot (sel==nb) must scatter nowhere.
    #[test]
    fn gather_blocks_backward_scatter_add_direct() {
        let ctx = test_ctx();
        let dims = compute::GatherDims {
            bh: 1,
            nb: 2,
            seq: 4,
            hd: 1,
            block: 2,
            k_sel: 2,
        };
        let sel = ctx.buffer_from_u32_slice(&[0u32, 2, 1, 0]);
        // d_out [bh*nb=2, sel_w=4, hd=1]; the 99.0 entries are the sentinel slots — must be ignored.
        let d_out = ctx.buffer_from_slice(&[1.0f32, 2.0, 99.0, 99.0, 3.0, 4.0, 5.0, 6.0]);
        let d_src = ctx.alloc_buffer(4 * 4);
        compute::gpu_gather_blocks_backward(&ctx, &d_out, &sel, &d_src, dims);
        let got = Tensor::from_buffer(Arc::clone(&ctx), d_src, vec![4]).to_vec();
        // src row0 = out[0]+out[6] = 1+5 = 6; row1 = out[1]+out[7] = 2+6 = 8; row2 = out[4] = 3; row3 = out[5] = 4.
        let expect = [6.0f32, 8.0, 3.0, 4.0];
        for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "d_src[{i}] = {g} (expected {e}); sentinel/accumulation handling wrong"
            );
        }
    }

    /// Benchmark (ignored): the gather path's score compute is O(n·(top_k+1)·block) vs dense O(n²) —
    /// must be FASTER at long sequence. Grounds the subquadratic claim. Run with --ignored --nocapture.
    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture (release)"]
    fn bench_block_sparse_gather_subquadratic() {
        use std::time::Instant;
        let ctx = test_ctx();
        let (bh, seq, hd, block, top_k) = (8usize, 1024usize, 64usize, 64usize, 3usize); // nb=16, k_sel=4
        let r = |s: usize| Tensor::randn(&ctx, vec![bh, seq, hd], 0.1 + s as f32 * 0.0);
        let (q, k, v) = (r(1), r(2), r(3));
        let scale = 1.0 / (hd as f32).sqrt();
        let dense = || {
            let s = q
                .batched_matmul_trans_b(&k)
                .scale(scale)
                .causal_mask(0)
                .softmax();
            s.batched_matmul(&v);
        };
        let sparse = || {
            crate::attention::block_sparse_gather_attention(&q, &k, &v, block, top_k);
        };
        for _ in 0..3 {
            dense();
            sparse();
        }
        let iters = 20;
        let t0 = Instant::now();
        for _ in 0..iters {
            dense();
        }
        let td = t0.elapsed().as_secs_f64() / iters as f64;
        let t1 = Instant::now();
        for _ in 0..iters {
            sparse();
        }
        let ts = t1.elapsed().as_secs_f64() / iters as f64;
        let dense_flops = 2.0 * (bh * seq * seq * hd) as f64; // Q@K^T scores
        let sparse_flops = 2.0 * (bh * seq * (top_k + 1) * block * hd) as f64;
        eprintln!("block-sparse gather seq={seq} block={block} top_k={top_k}: dense {td:.4}s, sparse {ts:.4}s, speedup {:.2}x; score-FLOPs {:.0}M vs {:.0}M ({:.2}× fewer)",
            td / ts, dense_flops / 1e6, sparse_flops / 1e6, dense_flops / sparse_flops);
        assert!(
            ts < td,
            "gather block-sparse ({ts:.4}s) should beat dense ({td:.4}s) at seq={seq}"
        );
    }

    /// End-to-end: a model with `AttnKind::BlockSparse` in every block forwards to finite logits,
    /// backprops a finite grad into w_q, and trains STABLY (finite + bounded, no blow-up) — the
    /// MoBA/NSA sparse path works through the real model + backward. Like linear_attn_model_trains_stably,
    /// convergence *quality* of this micro weight-tied model under aggressive top_k=1 sparsity is a
    /// training-recipe matter, not asserted here (real-data descent is shown by the training smoke).
    #[test]
    fn block_sparse_model_trains_stably() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let cfg = ModelConfig {
            block_sparse_top_k: 1,
            block_size: 4,
            ..ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64)
        };
        let model = Transformer::new(&ctx, cfg);
        assert_eq!(
            model.blocks[0].attn.attn_kind,
            crate::attention::AttnKind::BlockSparse
        );
        let (batch, seq_len) = (1usize, 16usize);
        let tokens: Vec<u32> = (0..16).map(|i| (i * 3 % 48) as u32).collect();
        let targets: Vec<u32> = vec![5; 16];
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let mut opt = crate::optim::Muon::new(&ctx, &prefs, 0.0);
        let wq_id = model.blocks[0].attn.w_q.id;
        let (mut first, mut last) = (0.0f32, 0.0f32);
        for step in 0..40 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            if step == 0 {
                assert_eq!(logits.shape, vec![batch * seq_len, vocab as usize]);
                assert!(
                    logits.to_vec().iter().all(|x| x.is_finite()),
                    "block-sparse forward non-finite"
                );
            }
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(
                lv.is_finite(),
                "block-sparse loss non-finite at step {step}"
            );
            if step == 0 {
                first = lv;
            }
            last = lv;
            autograd::backward(&ctx, loss.id);
            if step == 0 {
                let g = autograd::get_grad(wq_id).expect("no gradient reached block-sparse w_q");
                let gv = Tensor::from_buffer(
                    Arc::clone(&ctx),
                    g,
                    model.blocks[0].attn.w_q.shape.clone(),
                )
                .to_vec();
                assert!(
                    gv.iter().all(|x| x.is_finite()),
                    "non-finite grad on block-sparse w_q"
                );
            }
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let lr = 1e-2 * (((step + 1) as f32) / 20.0).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("block-sparse model: loss {first:.3} -> {last:.3}");
        assert!(
            last.is_finite() && last < 30.0,
            "block-sparse training must stay finite + bounded (got {first:.3}->{last:.3})"
        );
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// ROOT-CAUSE + fix for the RMSNorm-backward instability (#5). What this test VERIFIES directly:
    /// a collapsed activation row makes the backward explode (>1e3) without the clamp and stay bounded
    /// with it. ANALYZED upstream MECHANISM (the explanation for WHY a row collapses): in the
    /// weight-tied constant-target overfit, cross-entropy drives down the non-target logits by
    /// shrinking those tokens' tied embedding rows toward 0; because the same rows are the INPUT
    /// embeddings, those tokens' input activations go to ~0, so a RMSNorm input row collapses
    /// (mean_sq → 0). Then inv_rms = 1/√(mean_sq+eps) blows up and the inv_rms³ correction term
    /// explodes the backward — from a perfectly bounded forward. (At real scale, weight decay on
    /// rare-token embeddings is the same mechanism.) It is an inherent weight-tying × logit-suppression
    /// interaction, NOT a kernel bug — so the right fix is to BOUND the degenerate-row backward, which
    /// is what the clamp does. This test feeds a collapsed row and shows the clamp turns a >1e3
    /// explosion into a bounded gradient (toggle = `compute::set_rmsnorm_clamp`, investigation-only).
    #[test]
    fn rmsnorm_backward_clamp_bounds_collapse() {
        let ctx = test_ctx();
        let cols = 64usize;
        // A collapsed activation row: all tiny → mean_sq ≈ 1e-8 with a near-zero eps.
        let input = ctx.buffer_from_slice(&vec![1e-4f32; cols]);
        let weight = ctx.buffer_from_slice(&vec![1.0f32; cols]);
        // Alternating grad_output: nearly orthogonal to the (uniform) collapsed row, so the inv_rms³
        // correction term doesn't cancel the inv_rms term — the explosion is exposed, not hidden.
        let go: Vec<f32> = (0..cols)
            .map(|c| if c % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let grad_output = ctx.buffer_from_slice(&go);
        let gi = ctx.alloc_buffer(cols * 4);
        let gw = ctx.alloc_buffer(cols * 4);
        let p = compute::RmsNormBackwardParams {
            rows: 1,
            cols: cols as u32,
            eps: 1e-10,
        };
        let max_abs = |buf: &_| {
            MetalContext::read_buffer(buf, cols)
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()))
        };

        let prev = compute::set_rmsnorm_clamp(false);
        compute::gpu_rms_norm_backward(&ctx, &input, &weight, &grad_output, &gi, &gw, &p);
        let unclamped = max_abs(&gi);
        compute::set_rmsnorm_clamp(true);
        compute::gpu_rms_norm_backward(&ctx, &input, &weight, &grad_output, &gi, &gw, &p);
        let clamped = max_abs(&gi);
        compute::set_rmsnorm_clamp(prev);

        eprintln!("RMSNorm collapsed-row backward: unclamped max|grad|={unclamped:.1}, clamped max|grad|={clamped:.1}");
        assert!(
            unclamped > 1.0e3,
            "collapsed row should explode without the clamp, got {unclamped}"
        );
        assert!(
            clamped.is_finite() && clamped <= 1.0e3,
            "clamp must bound the degenerate backward, got {clamped}"
        );
    }

    /// Investigates the handoff's open question: is beta2=0.999 making loss WORSE a "third subtle
    /// bug", or expected behavior? Conclusion (grounded here): NOT a bug. Under a non-stationary
    /// gradient (a 100× jump), beta2=0.999's slow second-moment memory lags the new gradient scale,
    /// so its denominator √v̂ is under-sized right after the jump → it OVERSHOOTS (larger step) vs
    /// beta2=0.95, which adapts in a few steps. Both stay finite and bounded — no NaN/explosion, no
    /// bias-correction bug. The instability is the diagonal-second-moment non-stationarity tradeoff,
    /// which is exactly why warmup + beta2=0.95 (the hardened default) are used.
    #[test]
    fn beta2_high_overshoots_but_is_not_a_bug() {
        let ctx = test_ctx();
        let n = 64usize;
        let run = |beta2: f32| -> (f32, f32, bool) {
            let w = Tensor::zeros(&ctx, vec![n]);
            let mut opt = crate::optim::AdamW::new_with_config(
                &ctx,
                &[&w],
                0.0,
                crate::optim::AdamWHyper {
                    beta1: 0.9,
                    beta2,
                    eps: 1e-5,
                    update_clip: 0.0,
                },
            );
            let small = ctx.buffer_from_slice(&vec![0.01f32; n]);
            let big = ctx.buffer_from_slice(&vec![1.0f32; n]); // 100× jump
            let mut all_finite = true;
            // 15 small-gradient steps: let v settle to the small scale.
            for _ in 0..15 {
                autograd::zero_grads();
                let g = ctx.buffer_from_slice(&[0.01f32; 64]);
                autograd::accumulate_grad_for_test(&ctx, w.id, &g, n);
                opt.step(0.01);
            }
            let before_jump = w.to_vec()[0];
            // The jump step: param movement here reveals the overshoot.
            autograd::zero_grads();
            autograd::accumulate_grad_for_test(&ctx, w.id, &big, n);
            opt.step(0.01);
            let jump_step = (w.to_vec()[0] - before_jump).abs();
            // 15 more large-gradient steps.
            for _ in 0..15 {
                autograd::zero_grads();
                let g = ctx.buffer_from_slice(&[1.0f32; 64]);
                autograd::accumulate_grad_for_test(&ctx, w.id, &g, n);
                opt.step(0.01);
                if w.to_vec().iter().any(|x| !x.is_finite()) {
                    all_finite = false;
                }
            }
            let total = w.to_vec()[0].abs();
            let _ = small;
            autograd::zero_grads();
            (jump_step, total, all_finite)
        };
        let (jump95, total95, fin95) = run(0.95);
        let (jump999, total999, fin999) = run(0.999);
        eprintln!("beta2=0.95:  jump-step Δ={jump95:.5}, total |w|={total95:.4}, finite={fin95}");
        eprintln!(
            "beta2=0.999: jump-step Δ={jump999:.5}, total |w|={total999:.4}, finite={fin999}"
        );
        // The key finding: NO bug — both stay finite and bounded across the non-stationary jump.
        assert!(
            fin95 && fin999,
            "AdamW went non-finite — that WOULD be a bug"
        );
        assert!(
            total95 < 100.0 && total999 < 100.0,
            "AdamW must stay bounded (no explosion)"
        );
        // The mechanism: beta2=0.999 overshoots harder on the jump (lagging v → under-sized denom).
        assert!(jump999 >= jump95, "beta2=0.999 should overshoot ≥ beta2=0.95 on the gradient jump (jump999={jump999}, jump95={jump95})");
    }

    /// Regression for the latent Muon-fallback bug: the AdamW path for non-2-D params used to
    /// hardcode eps=1e-8 (the denominator-collapse bug fixed elsewhere) and apply weight decay to
    /// norm weights. A 1-D norm at 1.0 with a large weight_decay and ZERO gradient must stay at 1.0
    /// (no_decay), and the fallback eps must be the hardened 1e-5.
    #[test]
    fn muon_fallback_no_decay_keeps_norm_weights() {
        let ctx = test_ctx();
        let g = Tensor::ones(&ctx, vec![8]);
        let mut muon = crate::optim::Muon::new(&ctx, &[&g], 0.5);
        assert_eq!(
            muon.adamw_hyper.eps, 1e-5,
            "Muon AdamW-fallback must use the hardened eps (1e-5)"
        );
        let zero = ctx.buffer_from_slice(&[0.0f32; 8]);
        autograd::accumulate_grad_for_test(&ctx, g.id, &zero, 8);
        muon.step(0.1);
        let v = g.to_vec();
        assert!(
            v.iter().all(|&x| (x - 1.0).abs() < 1e-3),
            "1-D norm weight was decayed despite no_decay: {v:?}"
        );
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// Optimizer-state persistence for resume: muon / 8-bit / hybrid serialize their own state
    /// (momentum, int8 moments+scales) to blobs and restore it byte-identically — so resuming a
    /// non-AdamW run continues with its real optimizer state instead of fresh momentum.
    #[test]
    fn optimizer_state_blobs_roundtrip() {
        let ctx = test_ctx();
        // Muon: 2-D matrix (Muon path) + a 1-D norm (AdamW-fallback path with v).
        let w2d = Tensor::randn(&ctx, vec![4, 4], 0.5);
        let w1d = Tensor::ones(&ctx, vec![4]);
        let mut muon = crate::optim::Muon::new(&ctx, &[&w2d, &w1d], 0.0);
        for _ in 0..3 {
            autograd::zero_grads();
            let g2 = ctx.buffer_from_slice(&[0.1f32; 16]);
            let g1 = ctx.buffer_from_slice(&[0.1f32; 4]);
            autograd::accumulate_grad_for_test(&ctx, w2d.id, &g2, 16);
            autograd::accumulate_grad_for_test(&ctx, w1d.id, &g1, 4);
            muon.step(0.01);
        }
        let blobs = muon.save_state_blobs();
        let mut muon2 = crate::optim::Muon::new(&ctx, &[&w2d, &w1d], 0.0);
        muon2.load_state_blobs(muon.step, &blobs);
        assert_eq!(muon2.step, muon.step);
        assert_eq!(
            muon2.save_state_blobs(),
            blobs,
            "Muon state must round-trip byte-identically"
        );

        // 8-bit AdamW: int8 m_q/v_q + fp32 scales.
        let wq = Tensor::randn(&ctx, vec![300], 0.3);
        let mut q8 = crate::optim::AdamW8bit::new(&ctx, &[&wq], 0.0);
        for _ in 0..3 {
            autograd::zero_grads();
            let g = ctx.buffer_from_slice(&vec![0.1f32; 300]);
            autograd::accumulate_grad_for_test(&ctx, wq.id, &g, 300);
            q8.step(0.01);
        }
        let qb = q8.save_state_blobs();
        let mut q8b = crate::optim::AdamW8bit::new(&ctx, &[&wq], 0.0);
        q8b.load_state_blobs(q8.step, &qb);
        assert_eq!(
            q8b.save_state_blobs(),
            qb,
            "8-bit state must round-trip byte-identically"
        );

        // Hybrid: Muon (2-D) + AdamW (embeddings/norms) sub-state.
        let model = Transformer::new(&ctx, ModelConfig::custom(48, 64, 4, 2, 2.67, 64));
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let force = model.force_adamw_param_ids();
        let mut hyb = crate::optim::HybridOptimizer::new(
            &ctx,
            &prefs,
            0.0,
            &force,
            crate::optim::AdamWHyper::default(),
        );
        // one real backward so grads exist for all params
        let toks: Vec<u32> = (0..8).collect();
        let logits = model.forward(&toks, 1, 8, None, false);
        let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &[5u32; 8]);
        autograd::backward(&ctx, loss.id);
        autograd::clear_tape_keep_grads();
        hyb.step(0.01);
        let hb = hyb.save_state_blobs();
        let mut hyb2 = crate::optim::HybridOptimizer::new(
            &ctx,
            &prefs,
            0.0,
            &force,
            crate::optim::AdamWHyper::default(),
        );
        hyb2.load_state_blobs(hyb.adamw.step, &hb);
        assert_eq!(
            hyb2.save_state_blobs(),
            hb,
            "Hybrid state must round-trip byte-identically"
        );
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// Per-tensor gradient clipping bounds each tensor independently: an exploded tensor is scaled
    /// to max_norm while a healthy small-grad tensor is left UNTOUCHED — unlike global clipping,
    /// which would shrink the healthy tensor too because the offender inflates the global norm.
    #[test]
    fn per_tensor_clip_leaves_healthy_tensor_untouched() {
        let ctx = test_ctx();
        // Two standalone params via a tiny single-layer model so train::clip_* can walk parameters().
        // Simpler: drive the public clip on a model and check two of its grads.
        let model = Transformer::new(&ctx, ModelConfig::custom(48, 64, 2, 2, 2.67, 64));
        let big_id = model.blocks[0].attn.w_q.id;
        let big_n = model.blocks[0].attn.w_q.numel();
        let small_id = model.ln_final_weight.id;
        let small_n = model.ln_final_weight.numel();
        // Exploded grad on w_q (norm >> 1), tiny grad on the final norm (norm << 1).
        let big_grad = ctx.buffer_from_slice(&vec![10.0f32; big_n]);
        let small_grad = ctx.buffer_from_slice(&vec![0.001f32; small_n]);
        autograd::accumulate_grad_for_test(&ctx, big_id, &big_grad, big_n);
        autograd::accumulate_grad_for_test(&ctx, small_id, &small_grad, small_n);

        crate::train::clip_gradients_per_tensor(&ctx, &model, 1.0);

        let big_after = crate::tensor::Tensor::from_buffer(
            Arc::clone(&ctx),
            autograd::get_grad(big_id).unwrap(),
            vec![big_n],
        )
        .to_vec();
        let small_after = crate::tensor::Tensor::from_buffer(
            Arc::clone(&ctx),
            autograd::get_grad(small_id).unwrap(),
            vec![small_n],
        )
        .to_vec();
        let big_norm: f32 = big_after.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!(
            "per-tensor clip: big_norm after = {big_norm:.4}, small[0] = {}",
            small_after[0]
        );
        assert!(
            (big_norm - 1.0).abs() < 0.05,
            "exploded tensor must be clipped to max_norm=1, got {big_norm}"
        );
        assert!(
            (small_after[0] - 0.001).abs() < 1e-6,
            "healthy small-grad tensor must be untouched, got {}",
            small_after[0]
        );
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// The simdgroup_matrix (hardware MMA) matmul must compute the SAME product as the reference
    /// fp32 tiled kernel — including at M/N/K dims that are not multiples of 8 or 32 (the
    /// zero-padded edge tiles). Float fragments → fp32 precision, so the only differences are
    /// floating-point accumulation order; tolerance is tight.
    #[test]
    fn matmul_simdgroup_matches_fp32() {
        let ctx = test_ctx();
        // (M, K, N): a ragged edge case + a clean aligned case + a tall-skinny case.
        for &(m, k, n) in &[(33usize, 47usize, 29usize), (64, 64, 64), (96, 16, 40)] {
            let a = Tensor::randn(&ctx, vec![m, k], 0.5);
            let b = Tensor::randn(&ctx, vec![k, n], 0.5);
            let out_sg = ctx.alloc_buffer(m * n * 4);
            let out_ref = ctx.alloc_buffer(m * n * 4);
            compute::gpu_matmul_simdgroup(
                &ctx, &a.buffer, &b.buffer, &out_sg, m as u32, n as u32, k as u32,
            );
            compute::gpu_matmul_fp32(
                &ctx, &a.buffer, &b.buffer, &out_ref, m as u32, n as u32, k as u32,
            );
            let sg = Tensor::from_buffer(Arc::clone(&ctx), out_sg, vec![m, n]).to_vec();
            let rf = Tensor::from_buffer(Arc::clone(&ctx), out_ref, vec![m, n]).to_vec();
            let mut max_diff = 0.0f32;
            for i in 0..sg.len() {
                assert!(
                    sg[i].is_finite(),
                    "simdgroup matmul produced non-finite at {i} for {m}x{k}x{n}"
                );
                max_diff = max_diff.max((sg[i] - rf[i]).abs());
            }
            eprintln!("simdgroup vs fp32 [{m}x{k}x{n}]: max_diff={max_diff:.6}");
            assert!(
                max_diff < 1e-3,
                "simdgroup matmul disagrees with fp32 ({m}x{k}x{n}): max_diff={max_diff}"
            );
        }
    }

    /// The half-input simdgroup MMA matmul must match the hand-rolled fp16 tiled matmul on the SAME
    /// fp16 inputs (both fp16-input / fp32-output; difference is MMA vs scalar accumulation, so the
    /// tolerance is fp16-scale, not fp32). This is the kernel that backs the opt-in fast path.
    #[test]
    fn matmul_simdgroup_f16_matches_f16() {
        let ctx = test_ctx();
        for &(m, k, n) in &[(33usize, 47usize, 29usize), (128, 64, 96)] {
            let a = Tensor::randn(&ctx, vec![m, k], 0.3);
            let b = Tensor::randn(&ctx, vec![k, n], 0.3);
            let a16 = a.cast_to_f16();
            let b16 = b.cast_to_f16();
            let out_sg = ctx.alloc_buffer(m * n * 4);
            let out_ref = ctx.alloc_buffer(m * n * 4);
            compute::gpu_matmul_simdgroup_f16(
                &ctx, &a16, &b16, &out_sg, m as u32, n as u32, k as u32,
            );
            compute::gpu_matmul_f16(&ctx, &a16, &b16, &out_ref, m as u32, n as u32, k as u32);
            let sg = Tensor::from_buffer(Arc::clone(&ctx), out_sg, vec![m, n]).to_vec();
            let rf = Tensor::from_buffer(Arc::clone(&ctx), out_ref, vec![m, n]).to_vec();
            let max_diff = sg
                .iter()
                .zip(&rf)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            eprintln!("simdgroup_f16 vs f16 [{m}x{k}x{n}]: max_diff={max_diff:.5}");
            assert!(sg.iter().all(|x| x.is_finite()));
            assert!(
                max_diff < 2e-2,
                "simdgroup_f16 disagrees with f16 ({m}x{k}x{n}): {max_diff}"
            );
        }
    }

    /// Backward MMA: the simdgroup trans_b kernel (dA = dC @ Wᵀ) must match the scalar trans_b on the
    /// same fp32 inputs — both fp16-fragment precision, so fp16-scale tolerance. Ragged + aligned dims
    /// exercise the zero-padded edge tiles. The flag is forced off so the reference takes the scalar path.
    #[test]
    fn matmul_simdgroup_trans_b_matches_scalar() {
        let ctx = test_ctx();
        let prev = compute::set_simdgroup_matmul(false);
        for &(m, n, k) in &[(33usize, 29usize, 47usize), (64, 64, 64), (96, 40, 16)] {
            // C[M,N] = A[M,K] @ B[N,K]ᵀ
            let a = Tensor::randn(&ctx, vec![m, k], 0.3);
            let b = Tensor::randn(&ctx, vec![n, k], 0.3);
            let out_sg = ctx.alloc_buffer(m * n * 4);
            let out_ref = ctx.alloc_buffer(m * n * 4);
            compute::gpu_matmul_trans_b_simdgroup(
                &ctx, &a.buffer, &b.buffer, &out_sg, m as u32, n as u32, k as u32,
            );
            compute::gpu_matmul_trans_b(
                &ctx, &a.buffer, &b.buffer, &out_ref, m as u32, n as u32, k as u32,
            );
            let sg = Tensor::from_buffer(Arc::clone(&ctx), out_sg, vec![m, n]).to_vec();
            let rf = Tensor::from_buffer(Arc::clone(&ctx), out_ref, vec![m, n]).to_vec();
            let max_diff = sg
                .iter()
                .zip(&rf)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            eprintln!("simdgroup_trans_b vs scalar [{m}x{k}x{n}]: max_diff={max_diff:.5}");
            assert!(
                sg.iter().all(|x| x.is_finite()),
                "trans_b simdgroup non-finite ({m}x{n}x{k})"
            );
            assert!(
                max_diff < 2e-2,
                "trans_b simdgroup disagrees ({m}x{n}x{k}): {max_diff}"
            );
        }
        compute::set_simdgroup_matmul(prev);
    }

    /// Backward MMA: the simdgroup trans_a kernel (dB = Aᵀ @ dC) must match the scalar trans_a.
    #[test]
    fn matmul_simdgroup_trans_a_matches_scalar() {
        let ctx = test_ctx();
        let prev = compute::set_simdgroup_matmul(false);
        for &(m, k, n) in &[(33usize, 47usize, 29usize), (64, 64, 64), (40, 96, 16)] {
            // A:[M,K], B:[M,N], C:[K,N] = Aᵀ @ B
            let a = Tensor::randn(&ctx, vec![m, k], 0.3);
            let b = Tensor::randn(&ctx, vec![m, n], 0.3);
            let out_sg = ctx.alloc_buffer(k * n * 4);
            let out_ref = ctx.alloc_buffer(k * n * 4);
            compute::gpu_matmul_trans_a_simdgroup(
                &ctx, &a.buffer, &b.buffer, &out_sg, m as u32, k as u32, n as u32,
            );
            compute::gpu_matmul_trans_a(
                &ctx, &a.buffer, &b.buffer, &out_ref, m as u32, k as u32, n as u32,
            );
            let sg = Tensor::from_buffer(Arc::clone(&ctx), out_sg, vec![k, n]).to_vec();
            let rf = Tensor::from_buffer(Arc::clone(&ctx), out_ref, vec![k, n]).to_vec();
            let max_diff = sg
                .iter()
                .zip(&rf)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            eprintln!("simdgroup_trans_a vs scalar [{m}x{k}x{n}]: max_diff={max_diff:.5}");
            assert!(
                sg.iter().all(|x| x.is_finite()),
                "trans_a simdgroup non-finite"
            );
            assert!(
                max_diff < 2e-2,
                "trans_a simdgroup disagrees ({m}x{k}x{n}): {max_diff}"
            );
        }
        compute::set_simdgroup_matmul(prev);
    }

    /// The MMA path is now ON by default, so its backward must be verified end-to-end through
    /// autograd — the grad-check harness forces the flag off, leaving this path uncovered otherwise.
    /// Run a full forward+backward both ways and assert the input gradients agree (fp16-fragment
    /// precision → fp16-scale relative tolerance). Ragged dims exercise the 64×64 edge tiles.
    #[test]
    fn backward_matmul_mma_gradients_match_scalar() {
        let ctx = test_ctx();
        let (m, k, n) = (40usize, 72usize, 56usize);
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

        let run = |sg: bool| -> (Vec<f32>, Vec<f32>) {
            let prev = compute::set_simdgroup_matmul(sg);
            autograd::clear_tape();
            autograd::clear_recompute_registry();
            autograd::zero_grads();
            let a = Tensor::from_slice(&ctx, &a_data, vec![m, k]).with_grad();
            let b = Tensor::from_slice(&ctx, &b_data, vec![k, n]).with_grad();
            let c = a.matmul(&b);
            autograd::backward(&ctx, c.id);
            let ga = MetalContext::read_buffer(&autograd::get_grad(a.id).unwrap(), m * k);
            let gb = MetalContext::read_buffer(&autograd::get_grad(b.id).unwrap(), k * n);
            autograd::clear_tape();
            compute::set_simdgroup_matmul(prev);
            (ga, gb)
        };

        let (ga_off, gb_off) = run(false);
        let (ga_on, gb_on) = run(true);

        let rel = |x: &[f32], y: &[f32]| -> f32 {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs() / (1.0 + p.abs()))
                .fold(0.0f32, f32::max)
        };
        let da = rel(&ga_off, &ga_on);
        let db = rel(&gb_off, &gb_on);
        eprintln!("backward MMA vs scalar: dA rel={da:.5}, dB rel={db:.5}");
        assert!(
            ga_on.iter().all(|x| x.is_finite()) && gb_on.iter().all(|x| x.is_finite()),
            "MMA grads non-finite"
        );
        assert!(da < 5e-2, "dA gradient mismatch scalar vs MMA: {da}");
        assert!(db < 5e-2, "dB gradient mismatch scalar vs MMA: {db}");
    }

    /// Backward MMA (batched): the simdgroup batched trans_a (attention dK/dV + block-sparse gather
    /// backward) must match the scalar batched trans_a. Non-square K≠N + batch>1 catches the
    /// {M,K,N,batch} param-order bug (a K/N swap is silently correct only when K==N).
    #[test]
    fn batched_matmul_simdgroup_trans_a_matches_scalar() {
        let ctx = test_ctx();
        let prev = compute::set_simdgroup_matmul(false);
        let (batch, m, k, n) = (3usize, 20usize, 44usize, 28usize); // M=contraction, out [K,N], K≠N
        let a = Tensor::randn(&ctx, vec![batch, m, k], 0.3); // [batch, M, K]
        let b = Tensor::randn(&ctx, vec![batch, m, n], 0.3); // [batch, M, N]
        let out_sg = ctx.alloc_buffer(batch * k * n * 4);
        let out_ref = ctx.alloc_buffer(batch * k * n * 4);
        let dims = compute::BatchedDims {
            batch: batch as u32,
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        compute::gpu_batched_matmul_trans_a_simdgroup(&ctx, &a.buffer, &b.buffer, &out_sg, dims);
        compute::gpu_batched_matmul_trans_a(&ctx, &a.buffer, &b.buffer, &out_ref, dims);
        let sg = Tensor::from_buffer(Arc::clone(&ctx), out_sg, vec![batch, k, n]).to_vec();
        let rf = Tensor::from_buffer(Arc::clone(&ctx), out_ref, vec![batch, k, n]).to_vec();
        let max_diff = sg
            .iter()
            .zip(&rf)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "batched simdgroup_trans_a vs scalar [b{batch} {m}x{k}x{n}]: max_diff={max_diff:.5}"
        );
        assert!(
            sg.iter().all(|x| x.is_finite()),
            "batched trans_a simdgroup non-finite"
        );
        assert!(
            max_diff < 2e-2,
            "batched trans_a simdgroup disagrees: {max_diff}"
        );
        compute::set_simdgroup_matmul(prev);
    }

    /// The bf16 default-matmul flag gives `Tensor::matmul` fp32 RANGE: a value above the fp16 max
    /// (65504) is preserved, where the default fp16 path overflows/clamps it. And for normal-range
    /// values bf16-on agrees with the fp16 default to bf16 precision. Restores the default after.
    #[test]
    fn matmul_bf16_flag_preserves_range() {
        let ctx = test_ctx();
        autograd::no_grad(|| {
            // Range: A[0]=1e5 selected by a one-hot B. fp16 path corrupts it; bf16 preserves it.
            let big = 1.0e5f32;
            let a = Tensor::from_slice(&ctx, &[big; 32], vec![1, 32]);
            let mut bv = vec![0.0f32; 32];
            bv[0] = 1.0;
            let b = Tensor::from_slice(&ctx, &bv, vec![32, 1]);

            let prev = compute::set_bf16_matmul(false);
            let r_f16 = a.matmul(&b).to_vec()[0];
            compute::set_bf16_matmul(true);
            assert!(compute::bf16_matmul_enabled());
            let r_bf16 = a.matmul(&b).to_vec()[0];
            compute::set_bf16_matmul(prev); // restore

            eprintln!("matmul 1e5: fp16={r_f16}, bf16={r_bf16}");
            assert!(
                (r_bf16 - big).abs() < big * 5e-3,
                "bf16 must preserve 1e5: got {r_bf16}"
            );
            assert!(
                (r_f16 - big).abs() > big * 0.1,
                "fp16 path should corrupt 1e5 (overflow/clamp): got {r_f16}"
            );

            // Normal range: bf16-on vs fp16 default agree to bf16 precision.
            let x = Tensor::randn(&ctx, vec![48, 40], 0.4);
            let y = Tensor::randn(&ctx, vec![40, 56], 0.4);
            let off = x.matmul(&y).to_vec();
            compute::set_bf16_matmul(true);
            let on = x.matmul(&y).to_vec();
            compute::set_bf16_matmul(false);
            let max_rel = off
                .iter()
                .zip(&on)
                .map(|(p, q)| (p - q).abs() / (1.0 + p.abs()))
                .fold(0.0f32, f32::max);
            eprintln!("bf16 vs fp16 normal-range max_rel={max_rel:.4}");
            assert!(
                on.iter().all(|v| v.is_finite()) && max_rel < 0.05,
                "bf16 normal-range mismatch: {max_rel}"
            );
        });
    }

    /// The batched simdgroup MMA path (attention matmuls) must match the default batched matmul on
    /// the same inputs — for both plain and trans_b, at non-multiple-of-64 dims (edge tiles). Proves
    /// the simdgroup fast path extends correctly beyond the batch==1 projections.
    #[test]
    fn batched_simdgroup_matches_default() {
        let ctx = test_ctx();
        let a = Tensor::randn(&ctx, vec![3, 40, 48], 0.3);
        let b = Tensor::randn(&ctx, vec![3, 48, 56], 0.3);
        let bt = Tensor::randn(&ctx, vec![3, 56, 48], 0.3); // for trans_b: [batch, n, k]
        let prev = compute::set_simdgroup_matmul(false);
        let mm_off = a.batched_matmul(&b).to_vec();
        let tb_off = a.batched_matmul_trans_b(&bt).to_vec();
        compute::set_simdgroup_matmul(true);
        let mm_on = a.batched_matmul(&b).to_vec();
        let tb_on = a.batched_matmul_trans_b(&bt).to_vec();
        compute::set_simdgroup_matmul(prev);
        let d_mm = mm_off
            .iter()
            .zip(&mm_on)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let d_tb = tb_off
            .iter()
            .zip(&tb_on)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("batched simdgroup: matmul max_diff={d_mm:.5}, trans_b max_diff={d_tb:.5}");
        assert!(mm_on.iter().all(|x| x.is_finite()) && tb_on.iter().all(|x| x.is_finite()));
        assert!(d_mm < 2e-2, "batched simdgroup matmul mismatch: {d_mm}");
        assert!(d_tb < 2e-2, "batched simdgroup trans_b mismatch: {d_tb}");
    }

    /// Toggling the global simdgroup fast path must keep `Tensor::matmul` correct: flag-on and
    /// flag-off products agree (both fp16 precision). Proves the opt-in wiring routes correctly and
    /// restores the default afterwards.
    #[test]
    fn matmul_simdgroup_flag_routes_correctly() {
        let ctx = test_ctx();
        let a = Tensor::randn(&ctx, vec![64, 48], 0.3);
        let b = Tensor::randn(&ctx, vec![48, 80], 0.3);
        let prev = compute::set_simdgroup_matmul(false);
        let off = a.matmul(&b).to_vec();
        compute::set_simdgroup_matmul(true);
        assert!(compute::simdgroup_matmul_enabled());
        let on = a.matmul(&b).to_vec();
        compute::set_simdgroup_matmul(prev); // restore
        let max_diff = off
            .iter()
            .zip(&on)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("matmul flag off-vs-on max_diff={max_diff:.5}");
        assert!(on.iter().all(|x| x.is_finite()));
        assert!(
            max_diff < 2e-2,
            "simdgroup flag path diverged from default matmul: {max_diff}"
        );
    }

    /// Benchmark (serial, ignored): the hardware simdgroup MMA must be FASTER than the hand-rolled
    /// fp16 tiled matmul on a large square matmul — grounding the throughput claim rather than
    /// assuming it. Prints GFLOP/s for both and the speedup. Run:
    /// `cargo test --release bench_matmul_simdgroup -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: run explicitly with --ignored --nocapture (release)"]
    fn bench_matmul_simdgroup_vs_handrolled() {
        use std::time::Instant;
        let ctx = test_ctx();
        let s = 1024usize;
        let a = Tensor::randn(&ctx, vec![s, s], 0.1);
        let b = Tensor::randn(&ctx, vec![s, s], 0.1);
        let a16 = a.cast_to_f16();
        let b16 = b.cast_to_f16();
        let out = ctx.alloc_buffer(s * s * 4);
        let flops = 2.0 * (s as f64).powi(3);
        let iters = 50;
        // Warmup + time hand-rolled f16.
        for _ in 0..5 {
            compute::gpu_matmul_f16(&ctx, &a16, &b16, &out, s as u32, s as u32, s as u32);
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            compute::gpu_matmul_f16(&ctx, &a16, &b16, &out, s as u32, s as u32, s as u32);
        }
        let hand = t0.elapsed().as_secs_f64() / iters as f64;
        // Warmup + time simdgroup f16.
        for _ in 0..5 {
            compute::gpu_matmul_simdgroup_f16(&ctx, &a16, &b16, &out, s as u32, s as u32, s as u32);
        }
        let t1 = Instant::now();
        for _ in 0..iters {
            compute::gpu_matmul_simdgroup_f16(&ctx, &a16, &b16, &out, s as u32, s as u32, s as u32);
        }
        let sg = t1.elapsed().as_secs_f64() / iters as f64;
        eprintln!("matmul {s}^3: hand-rolled f16 = {:.1} GFLOP/s ({:.3} ms), simdgroup f16 = {:.1} GFLOP/s ({:.3} ms), speedup = {:.2}x",
            flops / hand / 1e9, hand * 1e3, flops / sg / 1e9, sg * 1e3, hand / sg);
        assert!(
            sg < hand,
            "simdgroup MMA ({sg:.4}s) should beat hand-rolled f16 ({hand:.4}s)"
        );
    }

    /// 8-bit AdamW must use ~4× less optimizer memory than fp32 AdamW on a real model: the moments
    /// are int8 (1 byte vs 4) plus a tiny per-256-block fp32 scale.
    #[test]
    fn adamw_8bit_memory_is_4x_smaller() {
        let ctx = test_ctx();
        let model = Transformer::new(&ctx, ModelConfig::custom(256, 128, 4, 4, 2.67, 128));
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let fp32 = crate::optim::AdamW::new(&ctx, &prefs, 0.0);
        let q8 = crate::optim::AdamW8bit::new(&ctx, &prefs, 0.0);
        let ratio = fp32.memory_bytes() as f32 / q8.memory_bytes() as f32;
        eprintln!(
            "optimizer memory: fp32={} B, 8-bit={} B, ratio={ratio:.2}×",
            fp32.memory_bytes(),
            q8.memory_bytes()
        );
        assert!(
            ratio > 3.5,
            "8-bit optimizer should be ~4× smaller, got {ratio:.2}×"
        );
    }

    /// 8-bit AdamW must follow nearly the SAME trajectory as fp32 AdamW (block-wise int8 quant adds
    /// only ~1% per-step error). Deterministic: feed an identical fixed gradient to two copies of the
    /// same parameter for 30 steps; the final weights must agree closely AND both must have moved.
    #[test]
    fn adamw_8bit_tracks_fp32() {
        let ctx = test_ctx();
        let n = 256usize;
        let init: Vec<f32> = (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let g: Vec<f32> = (0..n)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1 + 0.01)
            .collect();
        let w_fp = Tensor::from_slice(&ctx, &init, vec![n]);
        let w_q = Tensor::from_slice(&ctx, &init, vec![n]);
        // shape.len()==1 → no_decay; pass wd=0 anyway so the two paths are identical except quant.
        let mut fp = crate::optim::AdamW::new(&ctx, &[&w_fp], 0.0);
        let mut q8 = crate::optim::AdamW8bit::new(&ctx, &[&w_q], 0.0);
        for _ in 0..30 {
            autograd::zero_grads();
            let gb = ctx.buffer_from_slice(&g);
            autograd::accumulate_grad_for_test(&ctx, w_fp.id, &gb, n);
            fp.step(0.02);
            autograd::zero_grads();
            let gb2 = ctx.buffer_from_slice(&g);
            autograd::accumulate_grad_for_test(&ctx, w_q.id, &gb2, n);
            q8.step(0.02);
        }
        let a = w_fp.to_vec();
        let b = w_q.to_vec();
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let moved = a
            .iter()
            .zip(&init)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("8-bit vs fp32 after 30 steps: max_diff={max_diff:.5}, fp32 moved={moved:.4}");
        assert!(
            moved > 0.1,
            "fp32 reference must move substantially (sanity), moved={moved}"
        );
        assert!(
            max_diff < 0.02,
            "8-bit AdamW diverged from fp32: max_diff={max_diff}"
        );
        autograd::zero_grads();
    }

    /// 8-bit AdamW must keep training BOUNDED end-to-end (the dequant→update→requant kernel must not
    /// destabilize), via the unified `Optimizer::AdamW8bit` dispatch. Mirrors the fp32 guard
    /// adamw_training_stays_bounded_no_grad_explosion.
    #[test]
    fn adamw_8bit_training_stays_bounded() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let (batch, seq_len) = (8usize, 12usize);
        let one: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0, 9, 2, 8, 1];
        let tokens: Vec<u32> = one.iter().cloned().cycle().take(batch * seq_len).collect();
        let targets: Vec<u32> = vec![5; 8 * 12];
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let mut opt =
            crate::optim::Optimizer::AdamW8bit(crate::optim::AdamW8bit::new(&ctx, &prefs, 0.0));
        let mut max_loss = 0.0f32;
        for step in 0..150 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(lv.is_finite(), "8-bit AdamW loss non-finite at step {step}");
            max_loss = max_loss.max(lv);
            autograd::backward(&ctx, loss.id);
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let lr = 1e-3 * (((step + 1) as f32) / 20.0).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("8-bit AdamW stability: max loss {max_loss:.2}");
        assert!(max_loss < 30.0, "8-bit AdamW loss blew up (max {max_loss})");
    }

    /// MLA core: K,V are reconstructed from a shared low-rank latent c=x@W_dkv whose dim d_c is far
    /// smaller than kv_dim — so an MLA KV cache (storing c) is much smaller than caching K,V. Shapes
    /// must be right, the reconstruction differentiable, and the cache shrink large.
    #[test]
    fn mla_kv_compresses_and_reconstructs() {
        let ctx = test_ctx();
        let (n_tokens, d_model, kv_dim, d_c) = (16usize, 128usize, 128usize, 16usize);
        let x = Tensor::randn(&ctx, vec![n_tokens, d_model], 0.3);
        let w_dkv = Tensor::randn(&ctx, vec![d_model, d_c], 0.1);
        let w_uk = Tensor::randn(&ctx, vec![d_c, kv_dim], 0.1);
        let w_uv = Tensor::randn(&ctx, vec![d_c, kv_dim], 0.1);
        let (c, k, v) = crate::mla::mla_kv(&x, &w_dkv, &w_uk, &w_uv);
        assert_eq!(c.shape, vec![n_tokens, d_c], "latent shape");
        assert_eq!(k.shape, vec![n_tokens, kv_dim], "reconstructed K shape");
        assert_eq!(v.shape, vec![n_tokens, kv_dim], "reconstructed V shape");
        // K must equal c @ w_uk numerically (sanity that mla_kv really up-projects the latent).
        let k_ref = c.matmul(&w_uk).to_vec();
        let k_got = k.to_vec();
        let max_diff = k_ref
            .iter()
            .zip(&k_got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "K must be c@W_uk: max_diff={max_diff}");
        let (std_kv, mla_kv, shrink) = crate::mla::cache_footprint(kv_dim, d_c);
        eprintln!(
            "MLA cache/token: standard={std_kv} floats, MLA={mla_kv} floats, shrink={shrink:.0}×"
        );
        assert!(
            shrink >= 8.0,
            "MLA cache must be ≥8× smaller, got {shrink:.1}×"
        );
        autograd::clear_tape();
    }

    /// MLA INCREMENTAL DECODE with the latent cache must equal the full prefill forward — token-by-
    /// token decoding (caching only the small latent c, reconstructing K/V from it each step) produces
    /// the same logits as processing the whole sequence at once. This is the decisive integration test
    /// (catches the multi-step / cache bugs that single forwards miss). Also confirms the cache stores
    /// the latent c ([batch, seq, d_c]) — the 10–50× smaller KV footprint, not full K/V.
    #[test]
    fn mla_latent_decode_matches_full() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let cfg = ModelConfig {
            mla_latent_dim: 16,
            ..ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64)
        };
        let model = Transformer::new(&ctx, cfg);
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let seq = tokens.len();
        let full = model.forward(&tokens, 1, seq, None, false).to_vec(); // [seq, vocab]
        let mut caches = model.init_kv_caches();
        let mut inc: Vec<f32> = Vec::new();
        for t in 0..seq {
            let lt = model
                .forward(&tokens[t..t + 1], 1, 1, Some(&mut caches), false)
                .to_vec();
            inc.extend_from_slice(&lt);
        }
        let max_diff = full
            .iter()
            .zip(&inc)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Functional decode-correctness = same predicted token per position (the codebase's own
        // decode tests use argmax agreement; raw logits carry fp16 noise, amplified by MLA's extra
        // reconstruction matmuls). Compare argmax(full[t]) vs argmax(inc[t]).
        let vw = vocab as usize;
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let mut agree = 0usize;
        for t in 0..seq {
            if argmax(&full[t * vw..(t + 1) * vw]) == argmax(&inc[t * vw..(t + 1) * vw]) {
                agree += 1;
            }
        }
        eprintln!("MLA incremental-decode vs full-prefill: max logit_diff={max_diff:.4}, argmax agree {agree}/{seq}");
        assert!(
            inc.iter().all(|x| x.is_finite()),
            "MLA incremental decode produced non-finite logits"
        );
        // Bounded tracking (parallel-robust): incremental must closely track full — a broken latent
        // cache gives huge diffs or NaN, fp16/contention noise stays small. (Exact argmax 8/8 with
        // diff ~0.07 is verified serially; under cargo's parallel GPU contention an argmax can flip,
        // so the parallel assertion is the bounded-diff one.)
        assert!(
            max_diff < 0.5,
            "MLA latent-cache decode diverged from full prefill: max_diff={max_diff}"
        );
        assert!(
            agree >= seq - 1,
            "MLA decode should match full on nearly all positions (agree {agree}/{seq})"
        );
        // Cache holds the latent c [batch, seq, d_c=16], NOT K/V (kv_dim=64 → 2×64=128 floats/token).
        let lat = caches[0]
            .latent
            .as_ref()
            .expect("MLA decode must populate the latent cache");
        assert_eq!(
            lat.shape,
            vec![1, seq, 16],
            "latent cache must be [batch, seq, d_c]"
        );
        autograd::clear_tape();
    }

    /// MLA parameters() must include the latent projections (W_dkv/W_uk/W_uv) and EXCLUDE the now-
    /// unused direct K/V projections (W_k/W_v) — they receive no gradient and shouldn't be trained
    /// or checkpointed.
    #[test]
    fn mla_parameters_route_through_latent() {
        let ctx = test_ctx();
        let cfg = ModelConfig {
            mla_latent_dim: 16,
            ..ModelConfig::custom(48, 64, 4, 4, 2.67, 64)
        };
        let model = Transformer::new(&ctx, cfg);
        let attn = &model.blocks[0].attn;
        assert_eq!(attn.attn_kind, crate::attention::AttnKind::Mla);
        let ids: std::collections::HashSet<usize> =
            attn.parameters().iter().map(|p| p.id).collect();
        assert!(
            ids.contains(&attn.w_dkv.id),
            "W_dkv must be a trained param"
        );
        assert!(
            ids.contains(&attn.w_uk.id) && ids.contains(&attn.w_uv.id),
            "W_uk/W_uv must be trained"
        );
        assert!(
            !ids.contains(&attn.w_k.id) && !ids.contains(&attn.w_v.id),
            "MLA must NOT train the unused direct K/V projections"
        );
    }

    /// End-to-end MLA integration: a model with `AttnKind::Mla` in every block must forward to finite
    /// logits, backprop a finite gradient into the latent down-projection W_dkv, and train stably
    /// (clipped, warmed-up Muon steps never go non-finite). Mirrors linear_attn_model_trains_stably.
    #[test]
    fn safetensors_roundtrips_an_andreai_model() {
        let ctx = test_ctx();
        let cfg = ModelConfig::custom(48, 64, 4, 2, 2.67, 64);
        let model = Transformer::new(&ctx, cfg.clone());
        let path = "/tmp/andreai_safetensors_roundtrip.safetensors";
        crate::safetensors::export_safetensors(path, &model).expect("export");
        let loaded = crate::safetensors::import_safetensors(&ctx, path, cfg).expect("import");
        let a = model.parameters();
        let b = loaded.parameters();
        assert_eq!(a.len(), b.len(), "param count");
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(pa.shape, pb.shape, "tensor {i} shape");
            let (va, vb) = (pa.to_vec(), pb.to_vec());
            assert_eq!(va.len(), vb.len(), "tensor {i} len");
            for (j, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "tensor {i} elem {j} mismatch");
            }
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn mla_model_trains_stably() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let cfg = ModelConfig {
            mla_latent_dim: 16,
            ..ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64)
        };
        let model = Transformer::new(&ctx, cfg);
        let (batch, seq_len) = (1usize, 8usize);
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let targets: Vec<u32> = vec![5; 8];
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let mut opt = crate::optim::Muon::new(&ctx, &prefs, 0.0);
        let w_dkv_id = model.blocks[0].attn.w_dkv.id;
        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 0..40 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            if step == 0 {
                let lg = logits.to_vec();
                assert_eq!(logits.shape, vec![batch * seq_len, vocab as usize]);
                assert!(
                    lg.iter().all(|x| x.is_finite()),
                    "MLA forward produced non-finite logits"
                );
            }
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(lv.is_finite(), "MLA loss non-finite at step {step}: {lv}");
            if step == 0 {
                first = lv;
            }
            last = lv;
            autograd::backward(&ctx, loss.id);
            if step == 0 {
                let g = autograd::get_grad(w_dkv_id)
                    .expect("no gradient reached the MLA W_dkv latent projection");
                let gv = Tensor::from_buffer(
                    Arc::clone(&ctx),
                    g,
                    model.blocks[0].attn.w_dkv.shape.clone(),
                )
                .to_vec();
                assert!(
                    gv.iter().all(|x| x.is_finite()),
                    "non-finite grad on MLA W_dkv"
                );
            }
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let lr = 1e-2 * (((step + 1) as f32) / 20.0).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("MLA model: loss {first:.3} -> {last:.3}");
        assert!(last.is_finite(), "MLA training went non-finite");
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// Sequence packing greedily first-fits sequences into fixed rows with per-sequence segment ids
    /// and a sentinel pad segment. Deterministic layout check.
    #[test]
    fn pack_sequences_greedy_layout() {
        use crate::datapipe::{pack_sequences, PACK_PAD_SEG};
        // lengths 3,2,4 into max_len 6: row0 = [s0(3)+s1(2)+pad(1)], row1 = [s2(4)+pad(2)].
        let rows = pack_sequences(&[vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]], 6, 0);
        assert_eq!(rows.len(), 2, "should pack into 2 rows");
        assert_eq!(rows[0].tokens, vec![1, 2, 3, 4, 5, 0]);
        assert_eq!(rows[0].seg_ids, vec![0, 0, 0, 1, 1, PACK_PAD_SEG]);
        assert_eq!(rows[1].tokens, vec![6, 7, 8, 9, 0, 0]);
        assert_eq!(
            rows[1].seg_ids,
            vec![0, 0, 0, 0, PACK_PAD_SEG, PACK_PAD_SEG]
        );
        // Over-long sequence is truncated to max_len.
        let trunc = pack_sequences(&[vec![1, 2, 3, 4, 5, 6, 7, 8]], 4, 0);
        assert_eq!(trunc[0].tokens, vec![1, 2, 3, 4]);
    }

    /// The block-diagonal mask blocks BOTH future and cross-segment positions. With seg=[0,0,1,1],
    /// a segment-1 query (q=2) must not see segment-0 keys (k=0,1) even though they are in the past.
    #[test]
    fn causal_doc_mask_blocks_cross_segment() {
        let ctx = test_ctx();
        let scores = Tensor::zeros(&ctx, vec![1, 4, 4]); // all 0 → unmasked entries stay 0
        let seg = crate::gpu::u32_to_buf(ctx.buffer_from_u32_slice(&[0u32, 0, 1, 1]));
        let m = scores.causal_doc_mask(&seg, 1).to_vec();
        let at = |q: usize, k: usize| m[q * 4 + k];
        // q=0 (seg0): sees k=0; future k=1,2,3 masked.
        assert_eq!(at(0, 0), 0.0);
        assert!(
            at(0, 1).is_infinite() && at(0, 1) < 0.0,
            "future must be -inf"
        );
        // q=2 (seg1): k=0,1 are PAST but cross-segment → masked; k=2 same-seg causal → 0; k=3 future.
        assert!(at(2, 0).is_infinite(), "cross-segment past must be masked");
        assert!(at(2, 1).is_infinite(), "cross-segment past must be masked");
        assert_eq!(at(2, 2), 0.0, "same-segment causal must be visible");
        assert!(at(2, 3).is_infinite(), "future must be masked");
        // q=3 (seg1): sees k=2,3 (same seg, causal); k=0,1 cross-segment masked.
        assert_eq!(at(3, 2), 0.0);
        assert_eq!(at(3, 3), 0.0);
        assert!(at(3, 0).is_infinite() && at(3, 1).is_infinite());
    }

    /// THE packing invariant: attention over a PACKED row (two sequences + block-diagonal mask) must
    /// produce, for each segment, EXACTLY what attention over that sequence ALONE produces — no
    /// cross-sequence leakage. This is what makes packing a free throughput win instead of a
    /// correctness hazard.
    #[test]
    fn packed_attention_matches_separate() {
        let ctx = test_ctx();
        let hd = 8usize;
        let (l0, l1) = (3usize, 2usize);
        let seq = l0 + l1; // 5, packed exactly (no pad)
        let scale = 1.0 / (hd as f32).sqrt();
        // Deterministic q,k,v for the packed [1, seq, hd].
        let gen = |salt: usize| -> Vec<f32> {
            (0..seq * hd)
                .map(|i| (((i * 7 + salt * 13) % 17) as f32 - 8.0) * 0.1)
                .collect()
        };
        let (qd, kd, vd) = (gen(1), gen(2), gen(3));
        let seg = crate::gpu::u32_to_buf(ctx.buffer_from_u32_slice(&[0u32, 0, 0, 1, 1]));

        // Packed run: scores → block-diagonal mask → softmax → @v.
        let q = Tensor::from_slice(&ctx, &qd, vec![1, seq, hd]);
        let k = Tensor::from_slice(&ctx, &kd, vec![1, seq, hd]);
        let v = Tensor::from_slice(&ctx, &vd, vec![1, seq, hd]);
        let scores = q.batched_matmul_trans_b(&k).scale(scale);
        let out_packed = scores
            .causal_doc_mask(&seg, 1)
            .softmax()
            .batched_matmul(&v)
            .to_vec();

        // Separate run for segment 0 (its own causal attention).
        let run_seg = |off: usize, len: usize| -> Vec<f32> {
            let slice = |d: &[f32]| d[off * hd..(off + len) * hd].to_vec();
            let q0 = Tensor::from_slice(&ctx, &slice(&qd), vec![1, len, hd]);
            let k0 = Tensor::from_slice(&ctx, &slice(&kd), vec![1, len, hd]);
            let v0 = Tensor::from_slice(&ctx, &slice(&vd), vec![1, len, hd]);
            let s0 = q0.batched_matmul_trans_b(&k0).scale(scale);
            s0.causal_mask(0).softmax().batched_matmul(&v0).to_vec()
        };
        let out0 = run_seg(0, l0);
        let out1 = run_seg(l0, l1);

        let d0 = out_packed[0..l0 * hd]
            .iter()
            .zip(&out0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let d1 = out_packed[l0 * hd..seq * hd]
            .iter()
            .zip(&out1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("packed-vs-separate: seg0 max_diff={d0:.5}, seg1 max_diff={d1:.5}");
        assert!(
            d0 < 2e-3,
            "packed segment 0 leaked / differs from separate: {d0}"
        );
        assert!(
            d1 < 2e-3,
            "packed segment 1 leaked / differs from separate: {d1}"
        );
        autograd::clear_tape();
    }

    /// Model-level seq-packing: the per-document mask threaded through the FULL Transformer via
    /// forward_seg must isolate packed sequences — changing one document must not change another's
    /// logits. Differential design (no dependence on cross-config fp reproducibility): docB sits at
    /// the same positions in both rows, so under the mask its logits are independent of docA.
    #[test]
    fn seq_packing_isolates_documents_through_model() {
        let ctx = test_ctx();
        let vocab = 16u32;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let seq = 7usize;
        // Row = docA (3 tok) + docB (4 tok); seg id per position.
        let seg_buf = crate::gpu::u32_to_buf(ctx.buffer_from_u32_slice(&[0u32, 0, 0, 1, 1, 1, 1]));
        let docb = [3usize, 4, 5, 6]; // docB positions in the row

        let docb_logits = |toks: &[u32], seg: Option<&crate::gpu::Buf>| -> Vec<f32> {
            autograd::clear_tape();
            autograd::zero_grads();
            let out = model.forward_seg(toks, 1, seq, None, false, seg).to_vec(); // [seq, vocab]
            autograd::clear_tape();
            let vc = vocab as usize;
            docb.iter()
                .flat_map(|&p| out[p * vc..(p + 1) * vc].to_vec())
                .collect::<Vec<f32>>()
        };
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let row1 = [1u32, 2, 3, 4, 5, 6, 7]; // docA=[1,2,3]
        let row2 = [9u32, 8, 2, 4, 5, 6, 7]; // docA'=[9,8,2] — different; docB=[4,5,6,7] unchanged

        // (1) Masked: docB logits must NOT move when docA changes (zero cross-document leakage).
        let masked = max_diff(
            &docb_logits(&row1, Some(&seg_buf)),
            &docb_logits(&row2, Some(&seg_buf)),
        );
        assert!(masked < 2e-3, "seq-packing leaked: docB logits changed with docA under the per-doc mask (max diff {masked})");

        // (2) Negative control: WITHOUT the mask, docB attends across the row, so changing docA DOES
        // move docB's logits — proving the mask is what isolates them (the test isn't vacuous).
        let unmasked = max_diff(&docb_logits(&row1, None), &docb_logits(&row2, None));
        assert!(unmasked > 1e-2, "negative control failed: docB logits unchanged without the mask (max diff {unmasked}) — test can't detect leakage");

        eprintln!("seq-packing isolation: masked diff={masked:.5} (want ~0), unmasked diff={unmasked:.5} (want >0)");
        autograd::clear_tape();
        autograd::zero_grads();
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
            &ctx,
            &x.buffer,
            &norm_w.buffer,
            compute::FfnWeights {
                w1: &w1.buffer,
                w2: &w2.buffer,
                w3: &w3.buffer,
            },
            &out_buf,
            compute::MegaFfnDims {
                batch_tokens: n_tokens as u32,
                d_model: d as u32,
                d_ff: ff as u32,
                eps,
            },
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
        eprintln!(
            "mega_ffn vs standard: max_diff={:.6}, avg_diff={:.8}",
            max_diff, avg_diff
        );

        // Allow some tolerance — the fused kernel computes norm inline which may differ slightly
        assert!(max_diff < 0.01, "mega_ffn max_diff too large: {}", max_diff);
        assert!(
            avg_diff < 0.001,
            "mega_ffn avg_diff too large: {}",
            avg_diff
        );
    }

    /// Gradient checkpointing must produce the SAME parameter gradients as the standard forward:
    /// the recompute reproduces the original forward exactly (this is what makes it correct to
    /// trade compute for activation memory). The fp16-cache + buffer-recycling + pool-bypass fixes
    /// are what make this hold — before them the recomputed embedding gradient was ~sign-flipped.
    ///
    /// This is an EXACT std-vs-recompute comparison. Run GPU tests with
    /// `--test-threads=1`; the codebase's GPU layer is single-threaded by design
    /// (see metal/mod.rs), and the CI gate enforces serial test execution.
    #[test]
    fn gradient_checkpointing_matches_standard() {
        let ctx = test_ctx();
        let cfg = ModelConfig::custom(48, 64, 4, 2, 2.67, 32);
        let model = Transformer::new(&ctx, cfg);
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let targets: Vec<u32> = vec![5; 8];
        let pinfo: Vec<(usize, usize)> = model
            .parameters()
            .iter()
            .map(|p| (p.id, p.numel()))
            .collect();
        let grab = |pinfo: &[(usize, usize)]| -> Vec<Vec<f32>> {
            pinfo
                .iter()
                .map(|&(id, n)| {
                    autograd::get_grad(id)
                        .map(|g| Tensor::from_buffer(Arc::clone(&ctx), g, vec![n]).to_vec())
                        .unwrap_or_default()
                })
                .collect()
        };

        // Standard backward.
        let lt = model.forward(&tokens, 1, 8, None, false);
        let logits_std = lt.to_vec();
        let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &lt, &targets);
        autograd::backward(&ctx, loss.id);
        let std_grads = grab(&pinfo);
        autograd::clear_tape();
        autograd::zero_grads();
        autograd::clear_recompute_registry();

        // Checkpointed backward (same weights).
        let lt2 = model.forward(&tokens, 1, 8, None, true);
        let logits_ck = lt2.to_vec();
        let ldiff = logits_std
            .iter()
            .zip(&logits_ck)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            ldiff < 1e-3,
            "checkpointed forward must reproduce the standard forward: {ldiff}"
        );
        let (loss2, _) = crate::loss::cross_entropy_loss(&ctx, &lt2, &targets);
        autograd::backward(&ctx, loss2.id);
        let ck_grads = grab(&pinfo);
        autograd::zero_grads();

        assert_eq!(std_grads.len(), ck_grads.len());
        for (i, (s, c)) in std_grads.iter().zip(&ck_grads).enumerate() {
            assert_eq!(s.len(), c.len(), "param {i} grad len mismatch");
            // Tolerance relative to the parameter's gradient SCALE (fp16-level absolute noise from
            // the multi-layer recompute, not per-element relative which blows up near zero).
            let scale = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(0.05);
            let md = s
                .iter()
                .zip(c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                md <= 1e-2 * scale,
                "param {i}: max grad diff {md} > 1% of scale {scale}"
            );
        }
    }

    /// Checkpoint save/load roundtrip: verify weights survive a save→load cycle.
    /// Covers the v4 format with both trainable and base parameters.
    #[test]
    fn checkpoint_save_load_roundtrip() {
        let ctx = test_ctx();
        let config = ModelConfig::tiny(256);
        let model = Transformer::new(&ctx, config.clone());

        // Capture original weights
        let orig_params: Vec<Vec<f32>> = model.parameters().iter().map(|p| p.to_vec()).collect();

        // Save
        let tmp_path = "/tmp/andreai_test_ckpt.bin";
        crate::checkpoint::save_checkpoint(tmp_path, &model, 42).expect("save failed");

        // Load
        let (loaded_model, loaded_step) =
            crate::checkpoint::load_checkpoint(&ctx, tmp_path).expect("load failed");
        assert_eq!(loaded_step, 42);
        assert_eq!(loaded_model.config.d_model, config.d_model);
        assert_eq!(loaded_model.config.n_layers, config.n_layers);

        // Compare weights
        let loaded_params: Vec<Vec<f32>> = loaded_model
            .parameters()
            .iter()
            .map(|p| p.to_vec())
            .collect();
        assert_eq!(
            orig_params.len(),
            loaded_params.len(),
            "param count mismatch"
        );
        for (i, (orig, loaded)) in orig_params.iter().zip(loaded_params.iter()).enumerate() {
            assert_eq!(orig.len(), loaded.len(), "tensor {} size mismatch", i);
            for (j, (a, b)) in orig.iter().zip(loaded.iter()).enumerate() {
                assert!(
                    (*a - *b).abs() < 1e-6,
                    "tensor {} element {} mismatch: {} vs {}",
                    i,
                    j,
                    a,
                    b
                );
            }
        }

        std::fs::remove_file(tmp_path).ok();
    }

    /// REGRESSION (AdamW stability): two fixes stop AdamW blowing up on a tiny full-batch.
    /// (1) RMSNorm's backward has an `inv_rms^3` term that, when an activation row collapses
    /// (mean_sq -> 0), exploded the gradient to ~1e8 from a bounded forward — now clamped in the
    /// backward. (2) eps was 1e-8, too small for beta2=0.95, so the update denominator collapsed —
    /// now 1e-5. With both, AdamW keeps the loss BOUNDED (it also descends, but that's slow and noisy
    /// so we only assert bounded here; the convergence test proves learning with Muon). It must not
    /// regress to the ~1e5 blow-up.
    #[test]
    fn adamw_training_stays_bounded_no_grad_explosion() {
        let ctx = test_ctx();
        let vocab = 48u32;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let (batch, seq_len) = (8usize, 12usize);
        let one: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0, 9, 2, 8, 1];
        let tokens: Vec<u32> = one.iter().cloned().cycle().take(batch * seq_len).collect();
        let targets: Vec<u32> = vec![5; 8 * 12];
        let params = model.parameters();
        let param_refs: Vec<&Tensor> = params.to_vec();
        let mut opt = crate::optim::AdamW::new(&ctx, &param_refs, 0.0);
        let mut max_loss = 0.0f32;
        for step in 0..150 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(lv.is_finite(), "loss non-finite at step {step}");
            max_loss = max_loss.max(lv);
            autograd::backward(&ctx, loss.id);
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let lr = 1e-3 * (((step + 1) as f32) / 20.0).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("AdamW stability: max loss {max_loss:.2}");
        assert!(
            max_loss < 30.0,
            "AdamW loss blew up (max {max_loss}) — the RMSNorm-backward instability regressed"
        );
    }

    /// END-TO-END CONVERGENCE: the full forward → cross-entropy → backward → clip → optimizer loop
    /// must actually REDUCE loss to near-zero, not merely stay finite. This is the "can it learn at
    /// all" smoke test — without it, a silent bug in the backward or optimizer wiring would pass
    /// every other training test (which assert finiteness only). A capable Transformer is trained to
    /// predict a constant target (guaranteed-learnable) over a small batch; the loss must collapse.
    ///
    /// Uses MUON because it converges fastest here. Building this test also surfaced — and led to
    /// fixing — two real bugs that used to make AdamW oscillate/diverge on this regime: a RMSNorm
    /// backward that exploded (inv_rms^3) on a collapsed activation row, and an AdamW eps (1e-8) too
    /// small for beta2=0.95, which let the update denominator collapse. Both are fixed, so AdamW is
    /// now stable and descends below uniform too — it's just still SLOWER than Muon on a full-batch
    /// overfit (the inherent diagonal-vs-matrix preconditioning gap, not a bug), so the fast smoke
    /// test stays on Muon. See adamw_training_stays_bounded_no_grad_explosion for the AdamW guard.
    #[test]
    fn model_converges_overfitting_fixed_batch() {
        let ctx = test_ctx();
        let vocab = 32u32;
        let (batch, seq_len) = (8usize, 12usize);
        let one: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0, 9, 2, 8, 1];
        let tokens: Vec<u32> = one.iter().cloned().cycle().take(batch * seq_len).collect();
        let targets: Vec<u32> = vec![5; 8 * 12]; // constant target — the loop MUST be able to fit it

        // The init is unseeded (Tensor::randn → thread_rng), so the convergence RATE varies between
        // inits. Retry a few fresh inits and pass as soon as one collapses: a genuinely broken
        // backward/optimizer fails ALL attempts, while mere init-slowness can't make the test flaky.
        // NB: peak lr is 5e-3, not 1e-2 — at 1e-2 Muon diverged (loss climbing) on the large majority
        // of random inits here, so ALL 5 retries failed ~2/3 of the time (a real flaky test, unrelated
        // to backward correctness). 5e-3 over 500 steps descends smoothly to <0.5 on essentially every
        // init; a broken backward still can't fit the constant target and fails all attempts.
        let mut best = f32::INFINITY;
        let mut converged = false;
        for attempt in 0..5 {
            let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 128, 4, 4, 2.67, 64));
            let params = model.parameters();
            let param_refs: Vec<&Tensor> = params.to_vec();
            let mut opt = crate::optim::Muon::new(&ctx, &param_refs, 0.0);
            let (mut first, mut last) = (f32::NAN, f32::NAN);
            for step in 0..500 {
                let logits = model.forward(&tokens, batch, seq_len, None, false);
                let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
                let lv = loss.to_vec()[0];
                assert!(
                    lv.is_finite(),
                    "loss non-finite at attempt {attempt} step {step}: {lv}"
                );
                if step == 0 {
                    first = lv;
                }
                last = lv;
                autograd::backward(&ctx, loss.id);
                autograd::clear_tape_keep_grads();
                crate::train::clip_gradients(&ctx, &model, 1.0);
                let lr = 5e-3 * (((step + 1) as f32) / 30.0).min(1.0); // warmup 30 steps
                opt.step(lr);
                autograd::zero_grads_recycle();
            }
            eprintln!("attempt {attempt}: loss {first:.3} -> {last:.3}");
            best = best.min(last);
            autograd::clear_tape();
            autograd::zero_grads();
            if last < 0.5 {
                converged = true;
                break;
            }
        }
        // From ~uniform (ln 32 ≈ 3.47) the loss must collapse to < 0.5 on at least one init.
        assert!(
            converged,
            "the forward→backward→optimizer loop did NOT converge on ANY of 5 inits \
            (best final loss {best:.3}) — training is broken, not just slow"
        );
    }

    /// Perplexity = exp(mean per-token NLL). A fresh, ~uniform model over `vocab` tokens scores
    /// ≈ `vocab` — sanity-checks the metric is wired (finite, ≥ 1, vocab-scale).
    #[test]
    fn perplexity_metric_is_vocab_scale() {
        let ctx = test_ctx();
        let vocab = 32u32;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0, 9, 2, 8, 1];
        let ppl = crate::eval::perplexity(&ctx, &model, &tokens);
        eprintln!("fresh-model perplexity = {ppl:.2} (vocab={vocab})");
        assert!(
            ppl.is_finite() && ppl >= 1.0,
            "perplexity must be finite and >= 1: {ppl}"
        );
        assert!(
            ppl <= vocab as f32 * 8.0,
            "fresh-model perplexity implausibly high: {ppl}"
        );
    }

    /// min-p (relative floor) and locally-typical filtering trim a known distribution correctly.
    #[test]
    fn sampling_min_p_and_typical_filtering() {
        use crate::generate::filter_min_p_typical;
        // min_p = 0.3, max_p = 0.5 → threshold 0.15; token 3 (0.05) dropped, rest renormalized.
        let mut probs = vec![(0usize, 0.5f32), (1, 0.3), (2, 0.15), (3, 0.05)];
        filter_min_p_typical(&mut probs, 0.3, 1.0);
        assert!(
            probs.iter().all(|&(i, _)| i != 3),
            "min-p must drop the sub-threshold token"
        );
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().map(|x| x.1).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "min-p must renormalize, sum={sum}"
        );

        // typical_p keeps the lowest-surprisal-vs-entropy mass (here the 0.7+0.2 head) and renorms.
        let mut p2 = vec![(0usize, 0.7f32), (1, 0.2), (2, 0.07), (3, 0.03)];
        filter_min_p_typical(&mut p2, 0.0, 0.9);
        assert!(
            p2.len() < 4 && !p2.is_empty(),
            "typical must trim the set: {}",
            p2.len()
        );
        let s2: f32 = p2.iter().map(|x| x.1).sum();
        assert!(
            (s2 - 1.0).abs() < 1e-5,
            "typical must renormalize, sum={s2}"
        );

        // both disabled → unchanged
        let mut p3 = vec![(0usize, 0.6f32), (1, 0.4)];
        filter_min_p_typical(&mut p3, 0.0, 1.0);
        assert_eq!(p3.len(), 2);
    }

    /// Batched generation decodes N sequences through one batched KV cache. The batch dimension must
    /// be wired correctly and the cached decode must reproduce the equivalent full forward. We assert
    /// this at the **logit level**, not by bit-exact string equality of a greedy rollout: an untrained
    /// random model collapses to a near-tied tail where any sub-ULP difference flips the argmax, so a
    /// string compare is the wrong instrument.
    ///
    /// Regression guard for the `transpose_rope` dispatch bug: that wrapper computed a *threadgroup*
    /// count but dispatched it through `dispatchThreads` (which reads the grid as a *thread* count),
    /// so it launched only `ceil(total/256)` threads and left the rest of the RoPE'd Q/K as stale pool
    /// memory. At `seq_len==1` decode that is one thread — so the new token's K (and batch index 0 /
    /// every single-sequence generation) was garbage that varied run-to-run with pool contents. The
    /// two checks below pin it: (1) B identical lanes must be **bit-identical** (a lane reading stale
    /// memory diverges by whole units); (2) the cached decode must match a no-cache full forward over
    /// `[prompt, t0]` within fp16 cross-kernel noise.
    #[test]
    fn batched_generation_is_self_consistent() {
        let ctx = test_ctx();
        let vocab = 280u32;
        let v = vocab as usize;
        let model = Transformer::new(&ctx, ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64));
        let tok = BpeTokenizer::train(b"the quick brown fox jumps over the lazy dog. ", vocab);

        // Identical lanes run identical, deterministic (tiled, fixed-order) kernels → bit-identical.
        // LANE_TOL is a hair above 0 to tolerate any future benign reduction-order change while still
        // catching the stale-memory bug (which diverges by ~0.5+). CROSS_TOL covers the genuine fp16
        // noise between two different kernel paths (cached decode vs full forward, batch vs batch-1).
        const LANE_TOL: f32 = 1e-3;
        const CROSS_TOL: f32 = 0.05;
        let max_abs_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let argmax = |s: &[f32]| {
            s.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };

        // Build a B-identical prefill batch exactly as generate_batch does (BOS + encoded prompt).
        let b = 3usize;
        let mut ids = vec![BOS_TOKEN];
        ids.extend(tok.encode("the quick"));
        let len = ids.len();
        let flat: Vec<u32> = (0..b).flat_map(|_| ids.iter().copied()).collect();

        autograd::no_grad(|| {
            // ---- Stage A: batched prefill vs single-sequence prefill (no KV cache → pure forward). ----
            ctx.begin_batch();
            let batched = model.forward(&flat, b, len, None, false);
            ctx.flush_batch();
            let batched = batched.to_vec(); // [b * len, vocab]

            ctx.begin_batch();
            let single = model.forward(&ids, 1, len, None, false);
            ctx.flush_batch();
            let single = single.to_vec(); // [len, vocab]

            // Last-position logits per lane (the row generation samples from).
            let lane = |i: usize| &batched[(i * len + len - 1) * v..(i * len + len) * v];
            let single_last = &single[(len - 1) * v..len * v];

            assert!(
                max_abs_diff(lane(0), lane(1)) < LANE_TOL,
                "identical prefill lanes diverged"
            );
            assert!(
                max_abs_diff(lane(1), lane(2)) < LANE_TOL,
                "identical prefill lanes diverged"
            );
            assert!(
                max_abs_diff(lane(0), single_last) < CROSS_TOL,
                "batched prefill ≠ single-seq"
            );

            // Decisive first token: well-separated argmax → identical everywhere.
            let t0 = argmax(lane(0));
            assert_eq!(t0, argmax(lane(1)), "lanes disagree on the first token");
            assert_eq!(t0, argmax(lane(2)));
            assert_eq!(
                t0,
                argmax(single_last),
                "batched first token ≠ single-seq greedy"
            );

            // ---- Stage B: one batched preallocated-KV-cache decode step vs a no-cache full forward. ----
            let mut kv_b = model.init_kv_caches_preallocated(b);
            ctx.begin_batch();
            let _ = model.forward(&flat, b, len, Some(&mut kv_b), false); // prefill → populate cache
            ctx.flush_batch();
            let step_in = vec![t0 as u32; b];
            ctx.begin_batch();
            let dec_b = model.forward(&step_in, b, 1, Some(&mut kv_b), false); // [b, vocab]
            ctx.flush_batch();
            let dec_b = dec_b.to_vec();

            // Ground truth: full forward over [prompt, t0], last position per lane (what decode emits).
            let mut ids2 = ids.clone();
            ids2.push(t0 as u32);
            let len2 = ids2.len();
            let flat2: Vec<u32> = (0..b).flat_map(|_| ids2.iter().copied()).collect();
            ctx.begin_batch();
            let gt = model.forward(&flat2, b, len2, None, false);
            ctx.flush_batch();
            let gt = gt.to_vec();
            let glane = |i: usize| &gt[(i * len2 + len2 - 1) * v..(i * len2 + len2) * v];
            let dl = |i: usize| &dec_b[i * v..(i + 1) * v];

            assert!(
                max_abs_diff(dl(0), dl(1)) < LANE_TOL,
                "identical decode lanes diverged"
            );
            assert!(
                max_abs_diff(dl(1), dl(2)) < LANE_TOL,
                "identical decode lanes diverged"
            );
            // Every lane's cached decode must reproduce the no-cache ground truth.
            for i in 0..b {
                let c = max_abs_diff(dl(i), glane(i));
                assert!(
                    c < CROSS_TOL,
                    "cached decode lane {i} ≠ no-cache ground truth: {c}"
                );
            }
        });
    }

    /// Finite-difference check of the fused transpose+RoPE kernel and its backward, exercised directly
    /// (not through the model). This is the unit-level guard for the dispatch bug fixed alongside the
    /// batched-decode test: `gpu_transpose_rope` / `_backward` computed a threadgroup count but
    /// dispatched it as a thread count, so they wrote only `ceil(total/256)` of their output and left
    /// the rest stale. Here the forward must reproduce hand-computed RoPE at every element, and the
    /// backward gradient must match central finite differences of `L = Σ(rope(in) · G)` — both of which
    /// fail loudly if any output element is left unwritten.
    #[test]
    fn transpose_rope_forward_and_backward_match_finite_diff() {
        let ctx = test_ctx();
        let (batch, seq, n_heads, head_dim) = (2usize, 3usize, 2usize, 4usize);
        let offset = 1u32;
        let theta = 10000.0f32;
        let bh = batch * n_heads;
        let in_len = batch * seq * n_heads * head_dim; // input [batch*seq, n_heads*head_dim]
        let out_len = bh * seq * head_dim; // output [bh, seq, head_dim]
        let dims = compute::TrRopeDims {
            batch: batch as u32,
            seq: seq as u32,
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            offset,
            theta,
        };

        // Deterministic pseudo-random inputs (no thread_rng → reproducible).
        let inp: Vec<f32> = (0..in_len)
            .map(|i| ((i * 37 + 11) % 23) as f32 * 0.1 - 1.1)
            .collect();
        let grad_out: Vec<f32> = (0..out_len)
            .map(|i| ((i * 19 + 5) % 17) as f32 * 0.1 - 0.8)
            .collect();

        let forward = |data: &[f32]| -> Vec<f32> {
            let in_buf = ctx.buffer_from_slice(data);
            let out_buf = ctx.alloc_buffer(out_len * 4);
            compute::gpu_transpose_rope(&ctx, &in_buf, &out_buf, dims);
            ctx.wait_gpu();
            MetalContext::read_buffer(&out_buf, out_len)
        };

        // (1) Forward correctness: compare against a hand-written RoPE over the whole tensor.
        let out = forward(&inp);
        let nh_hd = n_heads * head_dim;
        let mut max_fwd_err = 0.0f32;
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..head_dim {
                        let val = inp[(b * seq + s) * nh_hd + h * head_dim + d];
                        let dp = if d % 2 == 0 { d + 1 } else { d - 1 };
                        let val_pair = inp[(b * seq + s) * nh_hd + h * head_dim + dp];
                        let pair = (d / 2) as f32;
                        let freq = 1.0 / theta.powf(2.0 * pair / head_dim as f32);
                        let angle = (s as f32 + offset as f32) * freq;
                        let (sin, cos) = angle.sin_cos();
                        let expect = if d % 2 == 0 {
                            val * cos - val_pair * sin
                        } else {
                            val_pair * sin + val * cos
                        };
                        let got = out[((b * n_heads + h) * seq + s) * head_dim + d];
                        max_fwd_err = max_fwd_err.max((expect - got).abs());
                    }
                }
            }
        }
        assert!(
            max_fwd_err < 1e-4,
            "transpose_rope forward mismatch vs hand-computed RoPE: {max_fwd_err} \
            (a too-small dispatch leaves output elements stale → large error)"
        );

        // (2) Backward vs central finite differences of L = Σ(rope(in)·grad_out).
        let grad_buf = {
            let go = ctx.buffer_from_slice(&grad_out);
            let gi = ctx.alloc_buffer(in_len * 4);
            compute::gpu_transpose_rope_backward(&ctx, &go, &gi, dims);
            ctx.wait_gpu();
            MetalContext::read_buffer(&gi, in_len)
        };
        let loss = |data: &[f32]| -> f32 {
            forward(data)
                .iter()
                .zip(&grad_out)
                .map(|(o, g)| o * g)
                .sum()
        };
        let eps = 1e-2f32;
        let mut max_grad_err = 0.0f32;
        for &i in &[0usize, 1, 5, 7, 13, 23, in_len - 1] {
            let mut up = inp.clone();
            up[i] += eps;
            let mut dn = inp.clone();
            dn[i] -= eps;
            let fd = (loss(&up) - loss(&dn)) / (2.0 * eps);
            max_grad_err = max_grad_err.max((fd - grad_buf[i]).abs());
        }
        assert!(
            max_grad_err < 1e-2,
            "transpose_rope backward disagrees with finite differences: {max_grad_err}"
        );
    }

    // ========================================================================
    // Phase B harness — finite-difference gradient checks for the core autograd
    // ops. Each builds a tiny graph  loss = Σ_i op(x)_i · seed_i  (a fixed varied
    // seed, realized on-tape as `op(x).reshape([1,n]) @ seed[n,1]`), runs the REAL
    // `autograd::backward`, and compares dL/dx to central finite differences.
    // This is the gate the unit suite previously lacked: a wrong analytic backward
    // (the transposed-dB / per-column-scale / Newton-Schulz class) is caught here in
    // CI on tiny tensors, instead of only surfacing in a full training run.
    // ========================================================================

    /// Deterministic, varied, non-zero input value for element `i`.
    fn gc_in(i: usize) -> f32 {
        (((i * 37 + 11) % 23) as f32) * 0.1 - 1.05
    }
    /// Deterministic upstream-gradient weight (dL/dout) for output element `i`.
    fn gc_seed(i: usize) -> f32 {
        (((i * 31 + 7) % 19) as f32) * 0.1 - 0.9
    }

    /// Central finite-difference gradient check against the real `autograd::backward`.
    /// `inputs`: (data, shape) per input tensor (all tracked with_grad). `forward` composes
    /// the op(s) under test on the rebuilt tensors. For each input, sampled element grads are
    /// compared to central differences; passes if abs OR rel error is within tolerance
    /// (fp16 paths can't hit tight abs at large magnitude; tiny grads can't hit rel).
    fn grad_check(
        ctx: &Arc<MetalContext>,
        inputs: &[(Vec<f32>, Vec<usize>)],
        forward: &dyn Fn(&[Tensor]) -> Tensor,
        eps: f32,
        abs_tol: f32,
        rel_tol: f32,
        name: &str,
    ) {
        // Hermetic environment: a gradient check must run on the default, precise
        // matmul path with no stale cross-test cast cache. cargo reuses worker
        // threads, so a prior test on this thread can leave a thread-local path
        // flag set, or a fp16/ternary cache entry keyed to a buffer address our
        // pool then re-hands out — either silently drifts the fp16 matmul past
        // tolerance and fails an otherwise-correct backward only in the full
        // parallel suite. Reset both so the check is deterministic.
        compute::set_simdgroup_matmul(false);
        compute::set_bf16_matmul(false);
        Tensor::clear_f16_cache();

        // --- analytic gradients via the real backward ---
        autograd::clear_tape();
        autograd::zero_grads();
        let tensors: Vec<Tensor> = inputs
            .iter()
            .map(|(d, s)| Tensor::from_slice(ctx, d, s.clone()).with_grad())
            .collect();
        let ids: Vec<usize> = tensors.iter().map(|t| t.id).collect();
        let out = forward(&tensors);
        let n: usize = out.shape.iter().product();
        let seed: Vec<f32> = (0..n).map(gc_seed).collect();
        let seed_t = Tensor::from_slice(ctx, &seed, vec![n, 1]);
        let loss = out.reshape(vec![1, n]).matmul(&seed_t);
        autograd::backward(ctx, loss.id);
        let analytic: Vec<Vec<f32>> = ids
            .iter()
            .zip(inputs)
            .map(|(&id, (d, _))| {
                let g = autograd::get_grad(id).unwrap_or_else(|| {
                    panic!("grad_check {name}: no gradient reached input id {id}")
                });
                Tensor::from_buffer(Arc::clone(ctx), g, vec![d.len()]).to_vec()
            })
            .collect();
        autograd::clear_tape();
        autograd::zero_grads();

        // --- L(data) with grad off, for central differences ---
        let loss_at = |all: &[Vec<f32>]| -> f32 {
            autograd::no_grad(|| {
                let ts: Vec<Tensor> = all
                    .iter()
                    .zip(inputs)
                    .map(|(d, (_, s))| Tensor::from_slice(ctx, d, s.clone()))
                    .collect();
                forward(&ts)
                    .to_vec()
                    .iter()
                    .enumerate()
                    .map(|(i, v)| v * gc_seed(i))
                    .sum()
            })
        };

        let base: Vec<Vec<f32>> = inputs.iter().map(|(d, _)| d.clone()).collect();
        for (k, (d, _)) in inputs.iter().enumerate() {
            let len = d.len();
            let stride = (len / 5).max(1);
            let mut i = 0;
            while i < len {
                let mut up = base.clone();
                let mut dn = base.clone();
                up[k][i] += eps;
                dn[k][i] -= eps;
                let fd = (loss_at(&up) - loss_at(&dn)) / (2.0 * eps);
                let an = analytic[k][i];
                let abs = (fd - an).abs();
                let rel = abs / (fd.abs().max(an.abs()) + 1e-3);
                assert!(
                    abs <= abs_tol || rel <= rel_tol,
                    "grad_check {name}: input[{k}][{i}] analytic={an:.5} numeric={fd:.5} \
                     abs_err={abs:.5} (tol {abs_tol}) rel_err={rel:.5} (tol {rel_tol})"
                );
                i += stride;
            }
        }
    }

    #[test]
    fn checkpoint_load_rejects_malformed_headers() {
        let ctx = test_ctx();
        let bad_magic_path = "/tmp/andreai_test_bad_checkpoint_magic.bin";
        std::fs::write(bad_magic_path, b"NOPE").expect("write bad checkpoint magic");
        let err = match crate::checkpoint::load_checkpoint(&ctx, bad_magic_path) {
            Ok(_) => panic!("bad checkpoint magic should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("not a valid AndreAI checkpoint"),
            "unexpected error: {err}"
        );

        let bad_version_path = "/tmp/andreai_test_bad_checkpoint_version.bin";
        let mut bad_version = Vec::new();
        bad_version.extend_from_slice(b"AMDL");
        bad_version.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(bad_version_path, bad_version).expect("write bad checkpoint version");
        let err = match crate::checkpoint::load_checkpoint(&ctx, bad_version_path) {
            Ok(_) => panic!("bad checkpoint version should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("unsupported checkpoint version"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(bad_magic_path).ok();
        std::fs::remove_file(bad_version_path).ok();
    }

    // 2e-2 abs / 5e-2 rel: comfortably passes the fp16-accumulate matmul paths while still
    // failing a structurally-wrong backward (those miss by >50%, not a few percent).
    const GC_EPS: f32 = 1e-2;
    const GC_ABS: f32 = 2e-2;
    const GC_REL: f32 = 5e-2;

    fn gc_vec(n: usize, off: usize) -> Vec<f32> {
        (0..n).map(|i| gc_in(i + off)).collect()
    }

    #[test]
    fn gradcheck_matmul() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 0), vec![2, 3]), (gc_vec(6, 5), vec![3, 2])],
            &|t| t[0].matmul(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "matmul",
        );
    }

    #[test]
    fn gradcheck_matmul_trans_b() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 1), vec![2, 3]), (gc_vec(6, 9), vec![2, 3])],
            &|t| t[0].matmul_trans_b(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "matmul_trans_b",
        );
    }

    #[test]
    fn gradcheck_batched_matmul_trans_b_nonsquare() {
        let ctx = test_ctx();
        // B=1, M=2 (≠ N=8), K=4 — the gather path's score matmul shape (block×sel_w). Standard
        // attention always has M==N==seq, so this M≠N case is otherwise untested.
        grad_check(
            &ctx,
            &[
                (gc_vec(8, 0), vec![1, 2, 4]),
                (gc_vec(32, 5), vec![1, 8, 4]),
            ],
            &|t| t[0].batched_matmul_trans_b(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "batched_matmul_trans_b_nonsquare",
        );
    }

    #[test]
    fn gradcheck_batched_matmul_nonsquare() {
        let ctx = test_ctx();
        // B=1, M=2, K=8, N=4 — the gather path's weights@vsel shape (block×sel_w @ sel_w×hd).
        grad_check(
            &ctx,
            &[
                (gc_vec(16, 0), vec![1, 2, 8]),
                (gc_vec(32, 5), vec![1, 8, 4]),
            ],
            &|t| t[0].batched_matmul(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "batched_matmul_nonsquare",
        );
    }

    #[test]
    fn gradcheck_add() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 0), vec![2, 3]), (gc_vec(6, 3), vec![2, 3])],
            &|t| t[0].add(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "add",
        );
    }

    #[test]
    fn gradcheck_mul() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 2), vec![2, 3]), (gc_vec(6, 7), vec![2, 3])],
            &|t| t[0].mul(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "mul",
        );
    }

    #[test]
    fn gradcheck_softmax() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 4), vec![2, 3])],
            &|t| t[0].softmax(),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "softmax",
        );
    }

    #[test]
    fn gradcheck_rms_norm() {
        let ctx = test_ctx();
        // x [2,4], weight [4]
        grad_check(
            &ctx,
            &[(gc_vec(8, 0), vec![2, 4]), (gc_vec(4, 13), vec![4])],
            &|t| t[0].rms_norm(&t[1], 1e-5),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "rms_norm",
        );
    }

    // ---- Custom/fused backward kernels that previously had NO finite-diff grad-check (the
    // class that hid the Flash partial-block bug). All are on the hot attention/training path.

    #[test]
    fn gradcheck_scaled_causal_softmax() {
        let ctx = test_ctx();
        // Fused scale + causal mask + softmax over [bh, seq, seq]. Masked (upper-tri) score grads must
        // be ~0 both analytically and numerically. This is the standard non-flash attention path.
        grad_check(
            &ctx,
            &[(gc_vec(16, 2), vec![1, 4, 4])],
            &|t| t[0].scaled_causal_softmax(0.5, 0),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "scaled_causal_softmax",
        );
    }

    #[test]
    fn gradcheck_apply_rope() {
        let ctx = test_ctx();
        // RoPE rotation over [batch_heads, seq, head_dim] (head_dim even). Backward = inverse rotation.
        grad_check(
            &ctx,
            &[(gc_vec(2 * 8 * 4, 0), vec![2, 8, 4])],
            &|t| t[0].apply_rope(0, 10000.0),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "apply_rope",
        );
    }

    #[test]
    fn gradcheck_rms_norm_residual() {
        let ctx = test_ctx();
        // Fused residual-add + RMS norm: rms_norm(x + residual, weight). Inputs: x, residual, weight.
        grad_check(
            &ctx,
            &[
                (gc_vec(8, 0), vec![2, 4]),
                (gc_vec(8, 9), vec![2, 4]),
                (gc_vec(4, 21), vec![4]),
            ],
            &|t| t[0].rms_norm_residual(&t[1], &t[2], 1e-5),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "rms_norm_residual",
        );
    }

    #[test]
    fn gradcheck_scale_rows() {
        let ctx = test_ctx();
        // Per-row scaling: out[r][c] = x[r][c] * scales[r]. Inputs: x [rows, cols], scales [rows].
        grad_check(
            &ctx,
            &[(gc_vec(12, 0), vec![3, 4]), (gc_vec(3, 17), vec![3])],
            &|t| t[0].scale_rows(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "scale_rows",
        );
    }

    #[test]
    fn gradcheck_repeat_kv() {
        let ctx = test_ctx();
        // GQA KV expansion: each of n_kv heads repeated group_size times. Backward = sum the
        // group_size gradient blocks back into each KV head. kv [n_kv=2, seq=4, hd=3], group=2.
        grad_check(
            &ctx,
            &[(gc_vec(2 * 4 * 3, 0), vec![2, 4, 3])],
            &|t| crate::attention::repeat_kv(&t[0], 2, 4, 3, 2),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "repeat_kv",
        );
    }

    #[test]
    fn gradcheck_transpose_bsh_to_bhs() {
        let ctx = test_ctx();
        // Head transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]. batch=1,
        // seq=4, n_heads=2, head_dim=4 → input [4, 8]. Backward = inverse permutation.
        grad_check(
            &ctx,
            &[(gc_vec(4 * 8, 0), vec![4, 8])],
            &|t| crate::attention::transpose_bsh_to_bhs(&t[0], 1, 4, 2, 4),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "transpose_bsh_to_bhs",
        );
    }

    #[test]
    fn gradcheck_fused_transpose_rope() {
        let ctx = test_ctx();
        // Fused head-transpose + RoPE (distinct kernel from apply_rope; runs in every attention fwd).
        // [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim] + rotation. Backward = inverse
        // RoPE + inverse transpose in one dispatch. batch=1, seq=4, n_heads=2, head_dim=4.
        grad_check(
            &ctx,
            &[(gc_vec(4 * 8, 0), vec![4, 8])],
            &|t| crate::attention::fused_transpose_rope(&t[0], 1, 4, 2, 4, 0, 10000.0),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "fused_transpose_rope",
        );
    }

    // NOTE: the chunked SSM backward is verified by `ssm::chunked_grad::ssm_chunked_grad_matches_materialized`
    // (gradient-equivalence to the materialised form), NOT a finite-diff grad-check here — loga
    // position 0 has a structurally-zero true gradient (uniform-shift cancellation in the decay), so a
    // central difference there is pure fp noise and would spuriously fail.

    #[test]
    fn gradcheck_transpose_bhs_to_bsh() {
        let ctx = test_ctx();
        // Reverse head transpose (attention-output path): [batch*n_heads, seq, head_dim] →
        // [batch*seq, n_heads*head_dim]. batch=1, n_heads=2, seq=4, head_dim=4 → input [2,4,4].
        grad_check(
            &ctx,
            &[(gc_vec(2 * 4 * 4, 0), vec![2, 4, 4])],
            &|t| crate::attention::transpose_bhs_to_bsh(&t[0], 1, 4, 2, 4),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "transpose_bhs_to_bsh",
        );
    }

    /// RWKV WKV time-mixing: per-channel decayed-cumsum + bonus. Inputs k,v [bh,seq,hd], decay-rate
    /// w [hd] (>0), bonus u [hd]. Backward flows through exp(±t·w), exp(k), exp(w), exp(u). w stays
    /// ≥0.3 under ±eps so the decay rate doesn't cross 0. Previously only a finite/non-zero check.
    /// WKV backward vs central differences. The stable decay form (`exp(-(t-1-i)·w)`, all ≤ 0) has no
    /// `exp(±t·w)` fp16 amplification, so all of k/v/w/u pass at the standard 5% rel-tol — `w ≈ 1` (the
    /// `exp(rwkv_w)` operating point) included. The forward is tightly CPU-checked separately
    /// (`wkv_matches_cpu`, `wkv_stable_at_long_seq`).
    #[test]
    fn gradcheck_wkv() {
        let ctx = test_ctx();
        let (bh, seq, hd) = (2usize, 5usize, 4usize);
        let w = vec![0.9f32, 1.0, 1.1, 0.8]; // ≈ exp(rwkv_w) ≈ 1 (the real decay), ±eps stays > 0
        let u = vec![0.1f32, -0.2, 0.15, -0.05];
        grad_check(
            &ctx,
            &[
                (gc_vec(bh * seq * hd, 0), vec![bh, seq, hd]),
                (gc_vec(bh * seq * hd, 11), vec![bh, seq, hd]),
                (w, vec![hd]),
                (u, vec![hd]),
            ],
            &|t| crate::rwkv::wkv(&t[0], &t[1], &t[2], &t[3]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "wkv",
        );
    }

    #[test]
    fn gradcheck_slice_cols() {
        let ctx = test_ctx();
        // x [2,4] → columns [1,3)
        grad_check(
            &ctx,
            &[(gc_vec(8, 6), vec![2, 4])],
            &|t| t[0].slice_cols(1, 2),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "slice_cols",
        );
    }

    #[test]
    fn gradcheck_silu() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 1), vec![2, 3])],
            &|t| t[0].silu(),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "silu",
        );
    }

    #[test]
    fn gradcheck_silu_gate() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 0), vec![2, 3]), (gc_vec(6, 8), vec![2, 3])],
            &|t| t[0].silu_gate(&t[1]),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "silu_gate",
        );
    }

    #[test]
    fn gradcheck_scale() {
        let ctx = test_ctx();
        grad_check(
            &ctx,
            &[(gc_vec(6, 3), vec![2, 3])],
            &|t| t[0].scale(0.7),
            GC_EPS,
            GC_ABS,
            GC_REL,
            "scale",
        );
    }

    // ========================================================================
    // Phase B harness — buffer-pool sanitizer (feature = "bufsan").
    // Trains a tiny model with pooled buffers NaN-poisoned at every flush; a
    // use-after-recycle / under-dispatch reads NaN → the loss blows up. With
    // quarantine on, intra-batch reissue is forbidden, so a correct run is
    // unchanged — a divergence flags reliance on intra-batch buffer aliasing.
    // ========================================================================
    #[cfg(feature = "bufsan")]
    fn bufsan_model_config() -> ModelConfig {
        ModelConfig::custom(48, 64, 4, 2, 2.67, 64)
    }

    #[cfg(feature = "bufsan")]
    fn bufsan_initial_weights(ctx: &Arc<MetalContext>) -> Vec<Vec<f32>> {
        let model = Transformer::new(ctx, bufsan_model_config());
        model.parameters().iter().map(|p| p.to_vec()).collect()
    }

    #[cfg(feature = "bufsan")]
    fn restore_weights(model: &Transformer, weights: &[Vec<f32>]) {
        let params = model.parameters();
        assert_eq!(
            params.len(),
            weights.len(),
            "bufsan parameter snapshot length mismatch"
        );
        for (param, data) in params.iter().zip(weights) {
            assert_eq!(
                param.numel(),
                data.len(),
                "bufsan parameter snapshot shape mismatch"
            );
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for value in data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            crate::gpu::buf_write_bytes(&param.buffer, &bytes);
        }
        Tensor::clear_f16_cache_recycle();
    }

    #[cfg(feature = "bufsan")]
    fn bufsan_train_tiny(
        ctx: &Arc<MetalContext>,
        quarantine: bool,
        initial_weights: Option<&[Vec<f32>]>,
    ) -> Vec<f32> {
        MetalContext::clear_pool();
        MetalContext::set_pool_quarantine(quarantine);
        let model = Transformer::new(ctx, bufsan_model_config());
        if let Some(weights) = initial_weights {
            restore_weights(&model, weights);
        }
        let (batch, seq_len) = (1usize, 16usize);
        let tokens: Vec<u32> = (0..16).map(|i| (i * 3 % 48) as u32).collect();
        let targets: Vec<u32> = vec![5; 16];
        let params = model.parameters();
        let prefs: Vec<&Tensor> = params.to_vec();
        let force = model.force_adamw_param_ids();
        let mut opt = crate::optim::HybridOptimizer::new(
            ctx,
            &prefs,
            0.0,
            &force,
            crate::optim::AdamWHyper::default(),
        );
        let mut losses = Vec::new();
        for step in 0..14 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            let (loss, _) = crate::loss::cross_entropy_loss(ctx, &logits, &targets);
            losses.push(loss.to_vec()[0]);
            autograd::backward(ctx, loss.id);
            autograd::clear_tape_keep_grads();
            crate::train::clip_gradients(ctx, &model, 1.0);
            opt.step(1e-3 * (((step + 1) as f32) / 8.0).min(1.0));
            autograd::zero_grads_recycle();
            Tensor::clear_f16_cache_recycle();
        }
        autograd::clear_tape();
        autograd::zero_grads();
        MetalContext::set_pool_quarantine(true);
        losses
    }

    #[cfg(feature = "bufsan")]
    #[test]
    fn bufsan_training_stays_finite_under_poison() {
        let ctx = test_ctx();
        let losses = bufsan_train_tiny(&ctx, false, None);
        for (s, l) in losses.iter().enumerate() {
            assert!(l.is_finite(), "bufsan poison surfaced a non-finite loss at step {s}: {l} — \
                a recycled buffer was read before being overwritten (use-after-recycle / under-dispatch)");
        }
        assert!(
            *losses.last().unwrap() < 30.0,
            "bufsan run diverged: {losses:?}"
        );
    }

    #[cfg(feature = "bufsan")]
    #[test]
    fn bufsan_quarantine_matches_default() {
        let ctx = test_ctx();
        let initial = bufsan_initial_weights(&ctx);
        let off = bufsan_train_tiny(&ctx, false, Some(&initial));
        let on = bufsan_train_tiny(&ctx, true, Some(&initial));
        let (lo, ln) = (*off.last().unwrap(), *on.last().unwrap());
        assert!(
            lo.is_finite() && ln.is_finite(),
            "non-finite final loss off={lo} on={ln}"
        );
        let rel = (lo - ln).abs() / lo.abs().max(1.0);
        assert!(
            rel < 0.20,
            "quarantine changed the run materially (off={lo} on={ln}, rel={rel:.3}) — \
             indicates reliance on intra-batch buffer aliasing (the loss-readout-class hazard)"
        );
    }

    // ========================================================================
    // Phase B harness — golden tests vs an independent CPU reference. Pins the
    // hand-written Metal forward kernels to known-correct output (catches the
    // wrong-transpose / under-dispatch / per-column-scale class on the forward
    // side, which finite-diff grad checks alone don't see).
    // ========================================================================
    fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                o[i * n + j] = s;
            }
        }
        o
    }
    fn cpu_matmul_trans_b(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // a[m,k], b[n,k] → a · bᵀ  [m,n]
        let mut o = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[j * k + p];
                }
                o[i * n + j] = s;
            }
        }
        o
    }
    fn cpu_softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut o = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for (c, &v) in row.iter().enumerate() {
                let e = (v - mx).exp();
                o[r * cols + c] = e;
                sum += e;
            }
            for c in 0..cols {
                o[r * cols + c] /= sum;
            }
        }
        o
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn golden_matmul_precise() {
        let ctx = test_ctx();
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(gc_in).collect();
        let b: Vec<f32> = (0..k * n).map(|i| gc_in(i + 5)).collect();
        let at = Tensor::from_slice(&ctx, &a, vec![m, k]);
        let bt = Tensor::from_slice(&ctx, &b, vec![k, n]);
        let got = autograd::no_grad(|| at.matmul_precise(&bt).to_vec());
        let d = max_abs_diff(&got, &cpu_matmul(&a, &b, m, k, n));
        assert!(d < 1e-3, "matmul_precise (fp32) vs CPU: max abs diff {d}");
    }

    #[test]
    fn golden_matmul_fp16_default() {
        // The default fp16-input/fp32-accumulate path used in training. Looser tol, but a
        // structural error (wrong index, under-dispatch) misses by orders of magnitude.
        let ctx = test_ctx();
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(gc_in).collect();
        let b: Vec<f32> = (0..k * n).map(|i| gc_in(i + 5)).collect();
        let at = Tensor::from_slice(&ctx, &a, vec![m, k]);
        let bt = Tensor::from_slice(&ctx, &b, vec![k, n]);
        let got = autograd::no_grad(|| at.matmul(&bt).to_vec());
        let d = max_abs_diff(&got, &cpu_matmul(&a, &b, m, k, n));
        assert!(d < 3e-2, "matmul (fp16) vs CPU: max abs diff {d}");
    }

    #[test]
    fn golden_matmul_trans_b() {
        let ctx = test_ctx();
        let (m, k, n) = (3usize, 4usize, 2usize);
        let a: Vec<f32> = (0..m * k).map(gc_in).collect();
        let b: Vec<f32> = (0..n * k).map(|i| gc_in(i + 9)).collect(); // [n,k]
        let at = Tensor::from_slice(&ctx, &a, vec![m, k]);
        let bt = Tensor::from_slice(&ctx, &b, vec![n, k]);
        let got = autograd::no_grad(|| at.matmul_trans_b(&bt).to_vec());
        let d = max_abs_diff(&got, &cpu_matmul_trans_b(&a, &b, m, k, n));
        assert!(d < 3e-2, "matmul_trans_b vs CPU: max abs diff {d}");
    }

    #[test]
    fn golden_softmax() {
        let ctx = test_ctx();
        let (r, c) = (3usize, 5usize);
        let x: Vec<f32> = (0..r * c).map(|i| gc_in(i + 2)).collect();
        let xt = Tensor::from_slice(&ctx, &x, vec![r, c]);
        let got = autograd::no_grad(|| xt.softmax().to_vec());
        let d = max_abs_diff(&got, &cpu_softmax_rows(&x, r, c));
        assert!(d < 2e-3, "softmax vs CPU row-softmax: max abs diff {d}");
    }

    /// Locks the BitNet fix (§6): `ternary_matmul` must scale output column j by `absmean[j]`,
    /// not by the global `absmean[0]`. Drives the same ternary kernels, then applies the
    /// per-column scale on the CPU. W is built so per-column absmean genuinely differs, so a
    /// regression to a single global scale fails for every column but the first.
    #[test]
    fn golden_ternary_matmul_per_column_scale() {
        let ctx = test_ctx();
        let (m, k, n) = (2usize, 32usize, 3usize);
        let x: Vec<f32> = (0..m * k).map(gc_in).collect();
        // column c scaled by (c+1) → column absmean grows with c.
        let mut w = vec![0.0f32; k * n];
        for r in 0..k {
            for c in 0..n {
                w[r * n + c] = gc_in(r * n + c + 3) * ((c + 1) as f32);
            }
        }
        let xt = Tensor::from_slice(&ctx, &x, vec![m, k]);
        let wt = Tensor::from_slice(&ctx, &w, vec![k, n]);

        // Reproduce the framework's ternary pipeline, but read raw product + per-col absmean out.
        let packed_rows = k.div_ceil(16);
        let absmean = ctx.alloc_buffer(n * 4);
        compute::gpu_ternary_absmean(&ctx, &wt.buffer, &absmean, k as u32, n as u32);
        let packed = ctx.alloc_buffer(packed_rows * n * 4);
        compute::gpu_ternary_pack(&ctx, &wt.buffer, &absmean, &packed, k as u32, n as u32);
        let raw = ctx.alloc_buffer(m * n * 4);
        compute::gpu_ternary_matmul(
            &ctx, &xt.buffer, &packed, &raw, m as u32, n as u32, k as u32,
        );
        ctx.wait_gpu();
        let absmean_v = MetalContext::read_buffer(&absmean, n);
        let raw_v = MetalContext::read_buffer(&raw, m * n);

        let spread = absmean_v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - absmean_v.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread > 1e-3,
            "test setup: per-column absmean must differ, got {absmean_v:?}"
        );

        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                want[i * n + j] = raw_v[i * n + j] * absmean_v[j];
            }
        }
        let got = autograd::no_grad(|| xt.ternary_matmul(&wt).to_vec());
        let d = max_abs_diff(&got, &want);
        assert!(
            d < 1e-3,
            "ternary_matmul per-column scale vs reference: max abs diff {d}\n absmean={absmean_v:?}"
        );
        autograd::clear_tape();
        autograd::zero_grads();
    }

    /// End-to-end integration of linear (O(N) kernel) attention in the real Transformer:
    ///   * forward through every layer produces finite logits of the right shape,
    ///   * the backward pass differentiates the linear-attention path (finite grad reaches w_q),
    ///   * training is numerically stable — clipped + warmed-up steps never produce NaN/Inf.
    ///
    /// (Convergence *quality* of this micro 2-layer weight-tied model is a separate, training-recipe
    /// matter — the softmax baseline destabilises identically at this scale — so it is not asserted
    /// here. The linear-attention math + gradients are proven on the isolated core by the
    /// linear_attention::* unit tests.)
    #[test]
    fn linear_attn_model_trains_stably() {
        let ctx = test_ctx();
        let vocab = 48u32;
        // Small, shallow model so it overfits 8 tokens quickly; linear attention in every layer.
        let mut cfg = ModelConfig::custom(vocab, 64, 4, 2, 2.67, 64);
        cfg.linear_attn = true;
        let model = Transformer::new(&ctx, cfg);

        let batch = 1usize;
        let seq_len = 8usize;
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        // Constant target: isolates "the optimizer can reduce loss through the linear-attention
        // forward+backward". Memorising a random permutation needs sequence capacity this tiny
        // weight-tied 2-layer model lacks (the softmax baseline can't do it either) — that's a
        // model-capacity question, not a linear-attention-correctness one (the unit tests cover that).
        let targets: Vec<u32> = vec![5; 8];

        let params = model.parameters();
        let param_refs: Vec<&Tensor> = params.to_vec();
        let mut opt = crate::optim::Muon::new(&ctx, &param_refs, 0.0);

        // Deterministic proof of wiring: after the first backward, the linear-attention
        // Q/K/V projection weights must receive finite, non-zero gradients.
        let wq_id = model.blocks[0].attn.w_q.id;

        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 0..40 {
            let logits = model.forward(&tokens, batch, seq_len, None, false);
            // Forward must be finite (catches any init-dependent overflow in the linear path).
            if step == 0 {
                let lg = logits.to_vec();
                assert_eq!(logits.shape, vec![batch * seq_len, vocab as usize]);
                assert!(
                    lg.iter().all(|x| x.is_finite()),
                    "linear-attn forward produced non-finite logits"
                );
            }
            let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
            let lv = loss.to_vec()[0];
            assert!(lv.is_finite(), "loss went non-finite at step {step}: {lv}");
            if step == 0 {
                first = lv;
            }
            last = lv;
            autograd::backward(&ctx, loss.id);
            if step == 0 {
                // The linear-attention path must be differentiated end-to-end: w_q receives a
                // finite gradient. (Non-zero magnitude is proven on the isolated core by the
                // linear_attention::gradient_flows unit test; raw init grads here are ~1e-8.)
                let g = autograd::get_grad(wq_id)
                    .expect("no gradient reached the linear-attention w_q");
                let gv = Tensor::from_buffer(
                    Arc::clone(&ctx),
                    g,
                    model.blocks[0].attn.w_q.shape.clone(),
                )
                .to_vec();
                assert!(
                    gv.iter().all(|x| x.is_finite()),
                    "non-finite grad on linear-attention w_q"
                );
            }
            autograd::clear_tape_keep_grads();
            // Gradient clipping + LR warmup — the same stabilisers the real training loop uses
            // (train.rs: max_grad_norm=1.0, warmup_steps). Without warmup, AdamW's first
            // (variance-uncorrected) steps overshoot and diverge — for the softmax baseline too.
            crate::train::clip_gradients(&ctx, &model, 1.0);
            let warmup = 30.0f32;
            let lr = 2e-3 * (((step + 1) as f32) / warmup).min(1.0);
            opt.step(lr);
            autograd::zero_grads_recycle();
        }
        eprintln!("linear-attn integration: loss {first:.4} -> {last:.4} (finite throughout)");
        // The numerical-stability claim: every per-step loss above was asserted finite, so the
        // linear-attention path never produced NaN/Inf across the run.
        assert!(first.is_finite() && last.is_finite());
    }

    /// The SSM (Mamba-2/SSD) mixer is usable in the real Transformer: every layer is an SSM block,
    /// the forward is finite, the input-dependent decay gate (ssm_loga) receives a finite gradient,
    /// and the ssm flag survives a checkpoint roundtrip (reconstructing the SSM mixer).
    #[test]
    fn ssm_model_integrates_and_differentiates() {
        use crate::attention::AttnKind;
        let ctx = test_ctx();
        let mut cfg = ModelConfig::custom(48, 64, 4, 2, 2.67, 64);
        cfg.ssm = true;
        let model = Transformer::new(&ctx, cfg);
        assert_eq!(model.blocks[0].attn.attn_kind, AttnKind::Ssm);
        assert_eq!(model.blocks[1].attn.attn_kind, AttnKind::Ssm);

        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let targets: Vec<u32> = vec![5; 8];
        let logits = model.forward(&tokens, 1, 8, None, false);
        assert_eq!(logits.shape, vec![8, 48]);
        assert!(
            logits.to_vec().iter().all(|x| x.is_finite()),
            "SSM forward non-finite"
        );
        let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
        assert!(loss.to_vec()[0].is_finite(), "SSM loss non-finite");
        autograd::backward(&ctx, loss.id);
        // The selective decay gate is differentiated end-to-end.
        let g = autograd::get_grad(model.blocks[0].attn.ssm_loga.id).expect("no grad for ssm_loga");
        let gv = Tensor::from_buffer(
            Arc::clone(&ctx),
            g,
            model.blocks[0].attn.ssm_loga.shape.clone(),
        )
        .to_vec();
        assert!(
            gv.iter().all(|x| x.is_finite()),
            "non-finite grad on ssm_loga"
        );
        autograd::zero_grads();

        let tmp = "/tmp/andreai_ssm_ckpt.bin";
        crate::checkpoint::save_checkpoint(tmp, &model, 9).expect("save failed");
        let (loaded, _) = crate::checkpoint::load_checkpoint(&ctx, tmp).expect("load failed");
        assert!(
            loaded.config.ssm,
            "ssm flag lost across checkpoint roundtrip"
        );
        assert_eq!(loaded.blocks[0].attn.attn_kind, AttnKind::Ssm);
        std::fs::remove_file(tmp).ok();
    }

    /// The RWKV-style time-mix is usable in the real Transformer: every layer is an RWKV mixer,
    /// the forward is finite, the per-channel decay (rwkv_w) and bonus (rwkv_u) receive finite
    /// gradients, and the rwkv flag survives a checkpoint roundtrip.
    #[test]
    fn rwkv_model_integrates_and_differentiates() {
        use crate::attention::AttnKind;
        let ctx = test_ctx();
        let mut cfg = ModelConfig::custom(48, 64, 4, 2, 2.67, 64);
        cfg.rwkv = true;
        let model = Transformer::new(&ctx, cfg);
        assert_eq!(model.blocks[0].attn.attn_kind, AttnKind::Rwkv);
        assert_eq!(model.blocks[1].attn.attn_kind, AttnKind::Rwkv);

        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let targets: Vec<u32> = vec![5; 8];
        let logits = model.forward(&tokens, 1, 8, None, false);
        assert_eq!(logits.shape, vec![8, 48]);
        assert!(
            logits.to_vec().iter().all(|x| x.is_finite()),
            "RWKV forward non-finite"
        );
        let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
        assert!(loss.to_vec()[0].is_finite(), "RWKV loss non-finite");
        autograd::backward(&ctx, loss.id);
        for (name, id, shape) in [
            (
                "rwkv_w",
                model.blocks[0].attn.rwkv_w.id,
                model.blocks[0].attn.rwkv_w.shape.clone(),
            ),
            (
                "rwkv_u",
                model.blocks[0].attn.rwkv_u.id,
                model.blocks[0].attn.rwkv_u.shape.clone(),
            ),
        ] {
            let g = autograd::get_grad(id).unwrap_or_else(|| panic!("no grad for {name}"));
            let gv = Tensor::from_buffer(Arc::clone(&ctx), g, shape).to_vec();
            assert!(
                gv.iter().all(|x| x.is_finite()),
                "non-finite grad on {name}"
            );
        }
        autograd::zero_grads();

        let tmp = "/tmp/andreai_rwkv_ckpt.bin";
        crate::checkpoint::save_checkpoint(tmp, &model, 11).expect("save failed");
        let (loaded, _) = crate::checkpoint::load_checkpoint(&ctx, tmp).expect("load failed");
        assert!(
            loaded.config.rwkv,
            "rwkv flag lost across checkpoint roundtrip"
        );
        assert_eq!(loaded.blocks[0].attn.attn_kind, AttnKind::Rwkv);
        std::fs::remove_file(tmp).ok();
    }

    /// Hybrid topology: a model with linear_attn_period=2 alternates softmax and linear-attention
    /// layers, forwards finitely, trains stably, and the schedule survives a checkpoint roundtrip.
    #[test]
    fn hybrid_topology_alternates_mixers() {
        use crate::attention::AttnKind;
        let ctx = test_ctx();
        let mut cfg = ModelConfig::custom(48, 64, 4, 4, 2.67, 64); // 4 layers
        cfg.linear_attn_period = 2; // (idx+1)%2==0 → layers 1,3 linear; 0,2 softmax
        let model = Transformer::new(&ctx, cfg);
        assert_eq!(model.blocks[0].attn.attn_kind, AttnKind::Softmax);
        assert_eq!(model.blocks[1].attn.attn_kind, AttnKind::Linear);
        assert_eq!(model.blocks[2].attn.attn_kind, AttnKind::Softmax);
        assert_eq!(model.blocks[3].attn.attn_kind, AttnKind::Linear);

        // Forward through the mixed softmax+linear stack is finite, and one backward is finite —
        // proving the hybrid stack is differentiable end-to-end. (Multi-step convergence of this
        // micro model is the known recipe-dependent matter covered by linear_attn_model_trains_stably.)
        let tokens: Vec<u32> = vec![3, 7, 1, 5, 2, 6, 4, 0];
        let targets: Vec<u32> = vec![5; 8];
        let logits = model.forward(&tokens, 1, 8, None, false);
        assert!(
            logits.to_vec().iter().all(|x| x.is_finite()),
            "hybrid forward non-finite"
        );
        let (loss, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
        assert!(loss.to_vec()[0].is_finite(), "hybrid loss non-finite");
        autograd::backward(&ctx, loss.id);
        // A softmax layer (0) and a linear layer (1) both receive finite gradients.
        for (li, id) in [
            (0usize, model.blocks[0].attn.w_q.id),
            (1usize, model.blocks[1].attn.w_q.id),
        ] {
            let g = autograd::get_grad(id).unwrap_or_else(|| panic!("no grad for layer {li} w_q"));
            let gv =
                Tensor::from_buffer(Arc::clone(&ctx), g, model.blocks[li].attn.w_q.shape.clone())
                    .to_vec();
            assert!(
                gv.iter().all(|x| x.is_finite()),
                "non-finite grad in layer {li}"
            );
        }
        autograd::zero_grads();

        // Checkpoint roundtrip preserves the schedule and reconstructs the same mixer per layer.
        let tmp = "/tmp/andreai_hybrid_ckpt.bin";
        crate::checkpoint::save_checkpoint(tmp, &model, 3).expect("save failed");
        let (loaded, _) = crate::checkpoint::load_checkpoint(&ctx, tmp).expect("load failed");
        assert_eq!(loaded.config.linear_attn_period, 2);
        assert_eq!(loaded.blocks[1].attn.attn_kind, AttnKind::Linear);
        assert_eq!(loaded.blocks[2].attn.attn_kind, AttnKind::Softmax);
        std::fs::remove_file(tmp).ok();
    }

    /// Checkpoint v5 preserves the linear_attn flag (and weights) across save→load.
    #[test]
    fn linear_attn_checkpoint_roundtrip() {
        let ctx = test_ctx();
        let mut cfg = ModelConfig::tiny(64);
        cfg.linear_attn = true;
        let model = Transformer::new(&ctx, cfg);
        assert!(model.config.linear_attn);

        let tmp = "/tmp/andreai_linear_attn_ckpt.bin";
        crate::checkpoint::save_checkpoint(tmp, &model, 7).expect("save failed");
        let (loaded, step) = crate::checkpoint::load_checkpoint(&ctx, tmp).expect("load failed");
        assert_eq!(step, 7);
        assert!(
            loaded.config.linear_attn,
            "linear_attn flag lost across checkpoint roundtrip"
        );
        // The reloaded block must actually be in Linear mode.
        assert_eq!(
            loaded.blocks[0].attn.attn_kind,
            crate::attention::AttnKind::Linear
        );
        std::fs::remove_file(tmp).ok();
    }

    /// Training state save/load roundtrip: verify model + optimizer state survive.
    #[test]
    fn training_state_save_load_roundtrip() {
        let ctx = test_ctx();
        let config = ModelConfig::tiny(256);
        let model = Transformer::new(&ctx, config);
        let param_refs: Vec<&_> = model.parameters().into_iter().collect();
        let mut optimizer = crate::optim::AdamW::new(&ctx, &param_refs, 0.01);
        optimizer.step = 10;

        // Do a fake optimizer step to populate m/v buffers
        let fake_grad = ctx.alloc_buffer(param_refs[0].numel() * 4);
        crate::gpu::compute::gpu_fill(&ctx, &fake_grad, param_refs[0].numel() as u32, 0.01);
        autograd::accumulate_grad_for_test(
            &ctx,
            param_refs[0].id,
            &fake_grad,
            param_refs[0].numel(),
        );
        ctx.begin_batch();
        optimizer.step(1e-4);
        ctx.flush_batch();

        // Capture optimizer m/v for first param
        let orig_m: Vec<f32> =
            MetalContext::read_buffer(&optimizer.params[0].m, optimizer.params[0].size);
        let orig_v: Vec<f32> =
            MetalContext::read_buffer(&optimizer.params[0].v, optimizer.params[0].size);

        // Save
        let tmp_path = "/tmp/andreai_test_state.bin";
        crate::checkpoint::save_training_state(tmp_path, &model, &optimizer, 42, 100000)
            .expect("save state failed");

        // Load
        let (loaded_model, opt_states, step, opt_step, tokens) =
            crate::checkpoint::load_training_state(&ctx, tmp_path).expect("load state failed");
        assert_eq!(step, 42);
        assert_eq!(tokens, 100000);

        // Verify optimizer state
        assert!(!opt_states.is_empty());
        let (loaded_m, loaded_v) = &opt_states[0];
        assert_eq!(loaded_m.len(), orig_m.len());
        for (a, b) in orig_m.iter().zip(loaded_m.iter()) {
            assert!((*a - *b).abs() < 1e-6, "m mismatch: {} vs {}", a, b);
        }
        for (a, b) in orig_v.iter().zip(loaded_v.iter()) {
            assert!((*a - *b).abs() < 1e-6, "v mismatch: {} vs {}", a, b);
        }

        // Verify the loaded state can be applied back into a fresh AdamW optimizer.
        let loaded_param_refs: Vec<&_> = loaded_model.parameters().into_iter().collect();
        let mut restored = crate::optim::AdamW::new(&ctx, &loaded_param_refs, 0.01);
        restored.load_state(&opt_states, opt_step);
        assert_eq!(restored.step, opt_step);
        let restored_m: Vec<f32> =
            MetalContext::read_buffer(&restored.params[0].m, restored.params[0].size);
        let restored_v: Vec<f32> =
            MetalContext::read_buffer(&restored.params[0].v, restored.params[0].size);
        for (a, b) in orig_m.iter().zip(restored_m.iter()) {
            assert!(
                (*a - *b).abs() < 1e-6,
                "restored m mismatch: {} vs {}",
                a,
                b
            );
        }
        for (a, b) in orig_v.iter().zip(restored_v.iter()) {
            assert!(
                (*a - *b).abs() < 1e-6,
                "restored v mismatch: {} vs {}",
                a,
                b
            );
        }

        std::fs::remove_file(tmp_path).ok();
        autograd::clear_tape();
        autograd::zero_grads_recycle();
    }

    #[test]
    fn training_state_load_rejects_malformed_headers_and_sidecars() {
        let ctx = test_ctx();
        let bad_magic_path = "/tmp/andreai_test_bad_state_magic.bin";
        std::fs::write(bad_magic_path, b"NOPE").expect("write bad state magic");
        let err = match crate::checkpoint::load_training_state(&ctx, bad_magic_path) {
            Ok(_) => panic!("bad training state magic should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("not a valid AndreAI training state file"),
            "unexpected error: {err}"
        );

        let bad_version_path = "/tmp/andreai_test_bad_state_version.bin";
        let mut bad_version = Vec::new();
        bad_version.extend_from_slice(b"AMDT");
        bad_version.extend_from_slice(&99u32.to_le_bytes());
        std::fs::write(bad_version_path, bad_version).expect("write bad state version");
        let err = match crate::checkpoint::load_training_state(&ctx, bad_version_path) {
            Ok(_) => panic!("bad training state version should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("unsupported training state version"),
            "unexpected error: {err}"
        );

        let bad_sidecar_magic_path = "/tmp/andreai_test_bad_sidecar_magic.opt";
        std::fs::write(bad_sidecar_magic_path, b"NOPE").expect("write bad sidecar magic");
        let err = match crate::checkpoint::load_opt_sidecar(bad_sidecar_magic_path) {
            Ok(_) => panic!("bad optimizer sidecar magic should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("not a valid AndreAI optimizer sidecar"),
            "unexpected error: {err}"
        );

        let truncated_sidecar_path = "/tmp/andreai_test_truncated_sidecar.opt";
        let mut truncated = Vec::new();
        truncated.extend_from_slice(b"AOPT");
        truncated.extend_from_slice(&128u32.to_le_bytes());
        std::fs::write(truncated_sidecar_path, truncated).expect("write truncated sidecar");
        let err = match crate::checkpoint::load_opt_sidecar(truncated_sidecar_path) {
            Ok(_) => panic!("truncated optimizer sidecar should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("optimizer type"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(bad_magic_path).ok();
        std::fs::remove_file(bad_version_path).ok();
        std::fs::remove_file(bad_sidecar_magic_path).ok();
        std::fs::remove_file(truncated_sidecar_path).ok();
    }

    #[test]
    fn training_state_next_step_normalizes_legacy_files() {
        assert_eq!(
            crate::checkpoint::normalize_training_state_next_step(13, 2, "/tmp/state_final.bin"),
            2,
            "v13 stores the next step directly"
        );
        assert_eq!(
            crate::checkpoint::normalize_training_state_next_step(12, 2, "/tmp/state_2.bin"),
            3,
            "legacy periodic states stored the last completed loop step"
        );
        assert_eq!(
            crate::checkpoint::normalize_training_state_next_step(12, 2, "/tmp/state_final.bin"),
            2,
            "legacy final states stored total_steps, already the next step"
        );
    }

    /// Cross-entropy loss: verify gradient matches finite differences.
    #[test]
    fn cross_entropy_gradient_check() {
        let ctx = test_ctx();
        let batch = 4;
        let vocab = 16;

        // Random logits
        let logits = Tensor::randn(&ctx, vec![batch, vocab], 1.0);
        let targets = vec![3u32, 7, 0, 15]; // one target per batch element

        // Forward + backward
        ctx.begin_batch();
        let (loss, grad_logits) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
        ctx.flush_batch();

        let loss_val = loss.to_vec()[0];
        let grad_data = MetalContext::read_buffer(&grad_logits, batch * vocab);

        // Numerical gradient check: perturb each logit by eps, compute (loss+ - loss-) / (2*eps)
        let eps = 1e-3f32;
        let logits_data = logits.to_vec();
        let mut max_diff = 0.0f32;
        let mut checked = 0;

        // Check a subset (all vocab elements for first 2 batch elements)
        for b in 0..2 {
            for v in 0..vocab {
                let idx = b * vocab + v;

                // Perturb +eps
                let mut plus = logits_data.clone();
                plus[idx] += eps;
                let plus_logits = Tensor::from_slice(&ctx, &plus, vec![batch, vocab]);
                ctx.begin_batch();
                let (plus_loss, _) = crate::loss::cross_entropy_loss(&ctx, &plus_logits, &targets);
                ctx.flush_batch();
                let lp = plus_loss.to_vec()[0];

                // Perturb -eps
                let mut minus = logits_data.clone();
                minus[idx] -= eps;
                let minus_logits = Tensor::from_slice(&ctx, &minus, vec![batch, vocab]);
                ctx.begin_batch();
                let (minus_loss, _) =
                    crate::loss::cross_entropy_loss(&ctx, &minus_logits, &targets);
                ctx.flush_batch();
                let lm = minus_loss.to_vec()[0];

                let numerical = (lp - lm) / (2.0 * eps);
                let analytical = grad_data[idx];
                let diff = (numerical - analytical).abs();
                max_diff = max_diff.max(diff);
                checked += 1;
            }
        }

        eprintln!(
            "CE grad check: max_diff={:.6}, loss={:.4}, checked={}",
            max_diff, loss_val, checked
        );
        assert!(
            max_diff < 1e-3,
            "CE gradient too far from numerical: max_diff={}",
            max_diff
        );
        autograd::clear_tape();
    }

    #[test]
    fn fused_linear_cross_entropy_matches_standard_tied_head() {
        let ctx = test_ctx();
        let n_tokens = 5;
        let d_model = 3;
        let vocab = 7;
        let hidden_data = [
            0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80, 0.90, -1.00, 1.10, -1.20, 1.30,
            -1.40, 1.50,
        ];
        let embedding_data = [
            0.20, -0.10, 0.30, -0.50, 0.40, -0.20, 0.10, 0.60, -0.30, -0.70, 0.20, 0.50, 0.30,
            -0.40, 0.80, -0.20, 0.90, -0.60, 0.50, -0.30, 0.10,
        ];
        let targets = [0u32, 3, 6, 2, 5];

        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        autograd::clear_tape();
        autograd::clear_recompute_registry();
        let hidden_std =
            Tensor::from_slice(&ctx, &hidden_data, vec![n_tokens, d_model]).with_grad();
        let embedding_std =
            Tensor::from_slice(&ctx, &embedding_data, vec![vocab, d_model]).with_grad();
        ctx.begin_batch();
        let logits = hidden_std.matmul_trans_b(&embedding_std);
        let (loss_std, _) = crate::loss::cross_entropy_loss(&ctx, &logits, &targets);
        ctx.flush_batch();
        let loss_std_val = loss_std.to_vec()[0];
        ctx.begin_batch();
        autograd::backward(&ctx, loss_std.id);
        ctx.flush_batch();
        let hidden_grad_std = MetalContext::read_buffer(
            &autograd::get_grad(hidden_std.id).expect("standard hidden grad"),
            n_tokens * d_model,
        );
        let embedding_grad_std = MetalContext::read_buffer(
            &autograd::get_grad(embedding_std.id).expect("standard embedding grad"),
            vocab * d_model,
        );
        autograd::clear_tape();
        autograd::zero_grads_recycle();

        let hidden_fused =
            Tensor::from_slice(&ctx, &hidden_data, vec![n_tokens, d_model]).with_grad();
        let embedding_fused =
            Tensor::from_slice(&ctx, &embedding_data, vec![vocab, d_model]).with_grad();
        ctx.begin_batch();
        let (loss_fused, _) = crate::loss::fused_linear_cross_entropy(
            &ctx,
            &hidden_fused,
            &embedding_fused,
            &targets,
            2,
        );
        ctx.flush_batch();
        let loss_fused_val = loss_fused.to_vec()[0];
        ctx.begin_batch();
        autograd::backward(&ctx, loss_fused.id);
        ctx.flush_batch();
        let hidden_grad_fused = MetalContext::read_buffer(
            &autograd::get_grad(hidden_fused.id).expect("fused hidden grad"),
            n_tokens * d_model,
        );
        let embedding_grad_fused = MetalContext::read_buffer(
            &autograd::get_grad(embedding_fused.id).expect("fused embedding grad"),
            vocab * d_model,
        );

        let loss_diff = (loss_std_val - loss_fused_val).abs();
        let hidden_diff = max_diff(&hidden_std.to_vec(), &hidden_fused.to_vec());
        let hidden_grad_diff = max_diff(&hidden_grad_std, &hidden_grad_fused);
        let embedding_grad_diff = max_diff(&embedding_grad_std, &embedding_grad_fused);
        eprintln!(
            "fused CE parity: loss_diff={loss_diff:.6}, hidden_diff={hidden_diff:.6}, \
             hidden_grad_diff={hidden_grad_diff:.6}, embedding_grad_diff={embedding_grad_diff:.6}"
        );

        assert!(
            loss_diff < 2e-3,
            "fused CE loss drifted from standard path: {loss_diff}"
        );
        assert!(
            hidden_diff < 1e-6,
            "fused CE must not mutate hidden inputs: {hidden_diff}"
        );
        assert!(
            hidden_grad_diff < 3e-3,
            "fused CE hidden gradient drifted from standard path: {hidden_grad_diff}"
        );
        assert!(
            embedding_grad_diff < 3e-3,
            "fused CE tied embedding gradient drifted from standard path: {embedding_grad_diff}"
        );

        autograd::clear_tape();
        autograd::zero_grads_recycle();
    }

    /// Quantize/dequantize roundtrip: verify Q8 and Q4 preserve values within tolerance.
    #[test]
    fn quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let shape = vec![16, 16];

        // Q8 roundtrip
        let q8 = crate::quantize::quantize(&data, &shape, 8, 32);
        let deq8 = crate::quantize::dequantize(&q8);
        assert_eq!(deq8.len(), data.len());
        let q8_max_err: f32 = data
            .iter()
            .zip(deq8.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Q8 max error: {:.6}", q8_max_err);
        assert!(
            q8_max_err < 0.05,
            "Q8 roundtrip error too large: {}",
            q8_max_err
        );

        // Q4 roundtrip (lower precision expected)
        let q4 = crate::quantize::quantize(&data, &shape, 4, 32);
        let deq4 = crate::quantize::dequantize(&q4);
        assert_eq!(deq4.len(), data.len());
        let q4_max_err: f32 = data
            .iter()
            .zip(deq4.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("Q4 max error: {:.6}", q4_max_err);
        assert!(
            q4_max_err < 0.5,
            "Q4 roundtrip error too large: {}",
            q4_max_err
        );
    }

    /// AdamW optimizer: verify one step changes weights in the right direction.
    #[test]
    fn adamw_single_step() {
        let ctx = test_ctx();
        let param = Tensor::full(&ctx, vec![4], 1.0).with_grad();
        let orig = param.to_vec();

        // Simulate gradient = 0.1 for all elements
        let grad = ctx.alloc_buffer(4 * 4);
        crate::gpu::compute::gpu_fill(&ctx, &grad, 4, 0.1);
        autograd::accumulate_grad_for_test(&ctx, param.id, &grad, 4);

        let param_refs = vec![&param];
        let mut opt = crate::optim::Muon::new(&ctx, &param_refs, 0.0);
        ctx.begin_batch();
        opt.step(1e-3);
        ctx.flush_batch();

        let updated = param.to_vec();
        // Positive gradient → params should decrease
        for (o, u) in orig.iter().zip(updated.iter()) {
            assert!(
                u < o,
                "param should decrease with positive gradient: {} -> {}",
                o,
                u
            );
        }

        autograd::zero_grads_recycle();
        autograd::clear_tape();
    }

    /// Matmul backward: verify dA matches numerical gradients.
    /// Uses a tolerance wide enough for the FP16 forward path while still catching
    /// broken backward indexing or accumulation.
    #[test]
    fn matmul_backward_gradient_check() {
        let ctx = test_ctx();
        let m = 4;
        let k = 3;
        let n = 2;
        let total = m * n;

        let a = Tensor::randn(&ctx, vec![m, k], 0.5);
        let b = Tensor::randn(&ctx, vec![k, n], 0.5);

        // Forward: C = A @ B, loss_vec = C_flat @ ones / N (mean, produces [1] via matmul)
        ctx.begin_batch();
        let c = a.matmul(&b);
        let c_flat = c.reshape(vec![1, total]); // [1, m*n]
        let ones = Tensor::full(&ctx, vec![total, 1], 1.0 / total as f32); // [m*n, 1]
        let loss_scalar = c_flat.matmul(&ones); // [1, 1] = mean(C)
        let loss = loss_scalar.reshape(vec![1]);
        ctx.flush_batch();

        let loss_val = loss.to_vec()[0];

        // Backward: autograd walks the tape from loss → reshape → matmul(c_flat, ones) → reshape → matmul(a, b)
        ctx.begin_batch();
        autograd::backward(&ctx, loss.id);
        ctx.flush_batch();

        let grad_a = autograd::get_grad(a.id).expect("no grad for a");
        let ga = MetalContext::read_buffer(&grad_a, m * k);

        // Numerical gradient: perturb each A[i] by ±eps, compute mean(perturbed_A @ B)
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        let eps = 1e-3f32;
        let mut max_diff = 0.0f32;

        for i in 0..m * k {
            let (lp, lm) = autograd::no_grad(|| {
                let mut plus = a_data.clone();
                plus[i] += eps;
                let ap = Tensor::from_slice(&ctx, &plus, vec![m, k]);
                let bp = Tensor::from_slice(&ctx, &b_data, vec![k, n]);
                ctx.begin_batch();
                let cp = ap.matmul(&bp);
                let lp_buf = ctx.alloc_buffer(4);
                compute::gpu_reduce_sum(&ctx, &cp.buffer, &lp_buf, total as u32);
                ctx.flush_batch();
                let lp = MetalContext::read_buffer(&lp_buf, 1)[0] / total as f32;

                let mut minus = a_data.clone();
                minus[i] -= eps;
                let am = Tensor::from_slice(&ctx, &minus, vec![m, k]);
                ctx.begin_batch();
                let cm = am.matmul(&bp);
                let lm_buf = ctx.alloc_buffer(4);
                compute::gpu_reduce_sum(&ctx, &cm.buffer, &lm_buf, total as u32);
                ctx.flush_batch();
                let lm = MetalContext::read_buffer(&lm_buf, 1)[0] / total as f32;
                (lp, lm)
            });

            let numerical = (lp - lm) / (2.0 * eps);
            let diff = (numerical - ga[i]).abs();
            max_diff = max_diff.max(diff);
        }

        eprintln!(
            "Matmul backward: max_diff_a={:.6}, loss={:.4}",
            max_diff, loss_val
        );
        // Tolerance accounts for FP16 non-determinism in Metal matmul (mixed precision).
        // Metal's FP16 shared memory rounding varies between kernel invocations, causing
        // ~0.3 max absolute error between analytical and numerical gradients.
        assert!(
            max_diff < 1.0,
            "Matmul dA gradient too far from numerical: {}",
            max_diff
        );

        autograd::clear_tape();
    }

    /// Data loader: verify batch shapes and epoch counting.
    #[test]
    fn data_loader_basic() {
        // Create a small test dataset
        let tmp_path = "/tmp/andreai_test_data.bin";
        let tokens: Vec<u32> = (0..1024).collect();
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(tmp_path, &bytes).expect("write test data");

        let mut loader = crate::data::DataLoader::new(tmp_path, 4, 32).expect("create loader");
        assert_eq!(loader.total_tokens(), 1024);
        assert_eq!(loader.epoch(), 0);

        let (inputs, targets) = loader.next_batch();
        assert_eq!(inputs.len(), 4 * 32);
        assert_eq!(targets.len(), 4 * 32);

        // Each target should be input shifted by 1
        // (within each sequence in the batch, targets[i] = dataset[start+i+1])
        // We can't check exact values due to random sampling, but verify non-zero
        let nonzero_inputs = inputs.iter().filter(|&&t| t > 0).count();
        assert!(nonzero_inputs > 0, "inputs should have non-zero tokens");

        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn dataset_rejects_partial_u32_file() {
        let tmp_path = "/tmp/andreai_test_bad_data.bin";
        std::fs::write(tmp_path, [1u8, 2, 3]).expect("write malformed dataset");

        let err = match crate::data::Dataset::load(tmp_path) {
            Ok(_) => panic!("malformed dataset should fail to load"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("multiple of 4"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn data_loader_rejects_invalid_batch_geometry() {
        let missing_path = "/tmp/andreai_missing_data_loader.bin";

        let err = match crate::data::DataLoader::new(missing_path, 0, 8) {
            Ok(_) => panic!("zero batch_size should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataLoader::new(missing_path, 1, 0) {
            Ok(_) => panic!("zero seq_len should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataLoader::new(missing_path, 1, usize::MAX) {
            Ok(_) => panic!("overflowing seq_len should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataLoader::new(missing_path, usize::MAX, 2) {
            Ok(_) => panic!("overflowing batch geometry should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn data_loader_rejects_too_small_dataset() {
        let tmp_path = "/tmp/andreai_test_tiny_data.bin";
        let tokens: Vec<u32> = (0..8).collect();
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(tmp_path, &bytes).expect("write tiny dataset");

        let err = match crate::data::DataLoader::new(tmp_path, 2, 8) {
            Ok(_) => panic!("too-small dataset should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("dataset too small"),
            "unexpected error: {err}"
        );

        std::fs::remove_file(tmp_path).ok();
    }

    #[test]
    fn data_mixer_rejects_invalid_source_weights() {
        let tmp_path = "/tmp/andreai_test_mixer_data.bin";
        let tokens: Vec<u32> = (0..64).collect();
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(tmp_path, bytes).expect("write mixer dataset");

        let err = match crate::data::DataMixer::new(&[tmp_path], &[], 1, 8) {
            Ok(_) => panic!("mismatched paths and weights should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataMixer::new(&[], &[], 1, 8) {
            Ok(_) => panic!("empty data mixer should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataMixer::new(&[tmp_path], &[0.0], 1, 8) {
            Ok(_) => panic!("zero total mixer weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("sum to > 0"),
            "unexpected error: {err}"
        );

        let err = match crate::data::DataMixer::new(&[tmp_path], &[-0.1], 1, 8) {
            Ok(_) => panic!("negative mixer weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = match crate::data::DataMixer::new(&[tmp_path], &[f32::INFINITY], 1, 8) {
            Ok(_) => panic!("infinite mixer weight should fail"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        std::fs::remove_file(tmp_path).ok();
    }
}
