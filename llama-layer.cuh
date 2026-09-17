#ifndef LLAMA_LAYER_CUH
#define LLAMA_LAYER_CUH

#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include "tlookup.cuh"
#include "timer.hpp"
#include <string>
#include <vector>

struct LayerDims {
    uint seq_len;
    uint embed_dim;
    uint hidden_dim;
    double eps;
};

// All parameters of one decoder layer, loaded from the files written by llama-commit.py.
// o_proj is deliberately absent: the upstream demo never proves it and the per-layer
// statement here mirrors upstream exactly.
struct LayerWeights {
    Weight input_norm, q, k, v, post_norm, up, gate, down;
    LayerWeights(const string& workdir, uint layer_idx, const LayerDims& d);
};

struct LayerTimers {
    double compute_s = 0.0;
    double prove_s = 0.0;
};

// The claim about a tensor X left by a sum-check: X evaluated at v (low-order variable first).
struct BoundaryClaim {
    vector<Fr_t> v;
    Fr_t value;
    bool valid = false;
};

FrTensor llama_rms_inv(const FrTensor& X, uint seq_len, uint embed_dim, double eps);

FrTensor llama_rmsnorm(const FrTensor& X, const Weight& w, const LayerDims& d, bool prove, LayerTimers& t, BoundaryClaim* input_claim_out);

FrTensor llama_attention(const FrTensor& X, const LayerWeights& w, const LayerDims& d, zkSoftmax& softmax, bool prove, LayerTimers& t);

FrTensor llama_ffn(const FrTensor& X, const LayerWeights& w, const LayerDims& d, tLookupRangeMapping& swiglu, bool prove, LayerTimers& t);

FrTensor run_layer(const FrTensor& x_in, const LayerDims& d, const LayerWeights& w, tLookupRangeMapping& swiglu, zkSoftmax& softmax,
    bool prove, LayerTimers& t, BoundaryClaim* input_claim_out);

#endif // LLAMA_LAYER_CUH
