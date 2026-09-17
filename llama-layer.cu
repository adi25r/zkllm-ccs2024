#include "llama-layer.cuh"

namespace {

// Adds the wall time of its scope to `acc`. Every kernel launch in this codebase is followed
// by cudaDeviceSynchronize, so host wall time is the device time.
class ScopedTimer {
    Timer tm;
    double& acc;
public:
    explicit ScopedTimer(double& a) : acc(a) { tm.start(); }
    ~ScopedTimer() { tm.stop(); acc += tm.getTotalTime(); }
};

template <typename F>
auto timed(double& acc, F&& f) -> decltype(f())
{
    ScopedTimer st(acc);
    return f();
}

Weight load_weight(const string& workdir, uint layer_idx, const string& name, uint in_dim, uint out_dim)
{
    const string prefix = "layer-" + std::to_string(layer_idx);
    return create_weight(
        workdir + "/" + name + "-pp.bin",
        workdir + "/" + prefix + "-" + name + "-int.bin",
        workdir + "/" + prefix + "-" + name + "-commitment.bin",
        in_dim, out_dim);
}

} // namespace

LayerWeights::LayerWeights(const string& workdir, uint layer_idx, const LayerDims& d):
    input_norm(load_weight(workdir, layer_idx, "input_layernorm.weight", 1, d.embed_dim)),
    q(load_weight(workdir, layer_idx, "self_attn.q_proj.weight", d.embed_dim, d.embed_dim)),
    k(load_weight(workdir, layer_idx, "self_attn.k_proj.weight", d.embed_dim, d.embed_dim)),
    v(load_weight(workdir, layer_idx, "self_attn.v_proj.weight", d.embed_dim, d.embed_dim)),
    post_norm(load_weight(workdir, layer_idx, "post_attention_layernorm.weight", 1, d.embed_dim)),
    up(load_weight(workdir, layer_idx, "mlp.up_proj.weight", d.embed_dim, d.hidden_dim)),
    gate(load_weight(workdir, layer_idx, "mlp.gate_proj.weight", d.embed_dim, d.hidden_dim)),
    down(load_weight(workdir, layer_idx, "mlp.down_proj.weight", d.hidden_dim, d.embed_dim))
{}

// One thread per row. Mirrors llama-rmsnorm.py: rms_inv = 1/sqrt(mean((X/sf)^2) + eps), stored as round(rms_inv*sf).
KERNEL void llama_layer_rms_inv_kernel(const Fr_t* X, Fr_t* out, uint rows, uint cols, double eps, double sf)
{
    const uint row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= rows) return;
    double acc = 0.0;
    for (uint j = 0; j < cols; ++j) {
        const double x = static_cast<double>(scalar_to_int(X[row * cols + j])) / sf;
        acc += x * x;
    }
    const double mean = acc / static_cast<double>(cols);
    out[row] = double_to_scalar(1.0 / sqrt(mean + eps), static_cast<unsigned long>(sf));
}

FrTensor llama_rms_inv(const FrTensor& X, uint seq_len, uint embed_dim, double eps)
{
    if (X.size != seq_len * embed_dim) throw std::runtime_error("llama_rms_inv: incompatible dimensions");
    FrTensor out(seq_len);
    llama_layer_rms_inv_kernel<<<(seq_len + FrNumThread - 1) / FrNumThread, FrNumThread>>>(
        X.gpu_data, out.gpu_data, seq_len, embed_dim, eps, 65536.0);
    cudaDeviceSynchronize();
    return out;
}

FrTensor llama_rmsnorm(const FrTensor& X, const Weight& w, const LayerDims& d, bool prove, LayerTimers& t, BoundaryClaim* input_claim_out)
{
    Rescaling rs1(1 << 16), rs2(1 << 16);
    zkFC g(1, d.embed_dim, w.weight);

    FrTensor rms_inv = timed(t.compute_s, [&] { return llama_rms_inv(X, d.seq_len, d.embed_dim, d.eps); });
    FrTensor g_inv_rms = timed(t.compute_s, [&] { return g(rms_inv); });
    FrTensor g_inv_rms_ = timed(t.compute_s, [&] { return rs1(g_inv_rms); });
    FrTensor Y = timed(t.compute_s, [&] { return g_inv_rms_ * X; });
    FrTensor Y_ = timed(t.compute_s, [&] { return rs2(Y); });

    if (prove) {
        timed(t.prove_s, [&] {
            rs2.prove(Y, Y_);
            auto u = random_vec(ceilLog2(Y.size));
            auto v = random_vec(ceilLog2(Y.size));
            auto pf = hadamard_product_sumcheck(g_inv_rms_, X, u, v);
            if (input_claim_out) {
                // Fr_hp_sc finishes with push_back(a(0)); push_back(b(0)); b is X folded by v.
                input_claim_out->v = v;
                input_claim_out->value = pf.back();
                input_claim_out->valid = true;
            }
            rs1.prove(g_inv_rms, g_inv_rms_);
            verifyWeightClaim(w, g.prove(rms_inv, g_inv_rms)[0]);
        });
    }
    return Y_;
}

FrTensor llama_attention(const FrTensor& X, const LayerWeights& w, const LayerDims& d, zkSoftmax& softmax, bool prove, LayerTimers& t)
{
    const uint seq_len = d.seq_len;
    const uint dim = d.embed_dim;   // single head with d = embed_dim, as in upstream self-attn.cu

    zkFC q_layer(dim, dim, w.q.weight);
    zkFC k_layer(dim, dim, w.k.weight);
    zkFC v_layer(dim, dim, w.v.weight);
    Rescaling q_rescale(1 << 16), k_rescale(1 << 16), v_rescale(1 << 16);

    FrTensor Q = timed(t.compute_s, [&] { return q_layer(X); });
    FrTensor Q_ = timed(t.compute_s, [&] { return q_rescale(Q); });
    FrTensor K = timed(t.compute_s, [&] { return k_layer(X); });
    FrTensor K_ = timed(t.compute_s, [&] { return k_rescale(K); });
    FrTensor V = timed(t.compute_s, [&] { return v_layer(X); });
    FrTensor V_ = timed(t.compute_s, [&] { return v_rescale(V); });

    if (prove) {
        timed(t.prove_s, [&] {
            q_rescale.prove(Q, Q_);
            k_rescale.prove(K, K_);
            v_rescale.prove(V, V_);
            verifyWeightClaim(w.k, k_layer.prove(X, K)[0]);
            verifyWeightClaim(w.q, q_layer.prove(X, Q)[0]);
            verifyWeightClaim(w.v, v_layer.prove(X, V)[0]);
        });
    }

    Rescaling rs1(1 << 20), rs2(1 << 20);
    FrTensor shift(seq_len), S_shifted(seq_len * seq_len);
    vector<FrTensor> S_segments, Y_segments, m_segments;

    FrTensor S = timed(t.compute_s, [&] { return FrTensor::matmul(Q_, K_.transpose(seq_len, dim), seq_len, dim, seq_len); });
    FrTensor Y = timed(t.compute_s, [&] { return softmax.compute(S, shift, S_shifted, S_segments, Y_segments, m_segments); });
    FrTensor out = timed(t.compute_s, [&] { return FrTensor::matmul(Y, V_, seq_len, seq_len, dim); });
    FrTensor out_ = timed(t.compute_s, [&] { return rs2(out); });
    FrTensor out__ = timed(t.compute_s, [&] { return rs1(out_); });

    if (prove) {
        timed(t.prove_s, [&] {
            rs1.prove(out_, out__);
            rs2.prove(out, out_);
            auto temp_rand = random_vec(3);
            vector<Polynomial> proof;
            auto u1 = random_vec(ceilLog2(seq_len));
            auto u2 = random_vec(ceilLog2(dim));
            auto ud = random_vec(ceilLog2(seq_len));
            auto claim = out.multi_dim_me({u1, u2}, {seq_len, dim});
            zkip(claim, Y.partial_me(u1, seq_len, seq_len), V_.partial_me(u2, dim, 1), ud, proof);

            softmax.prove(Y, S, shift, S_shifted, S_segments, Y_segments, m_segments,
                random_vec(ceilLog2(Y.size)), random_vec(ceilLog2(Y.size)), temp_rand[0], temp_rand[1], temp_rand[2], proof);

            auto u1_ = random_vec(ceilLog2(seq_len));
            auto u2_ = random_vec(ceilLog2(seq_len));
            auto ud_ = random_vec(ceilLog2(dim));
            auto claim_ = S.multi_dim_me({u1_, u2_}, {seq_len, seq_len});
            zkip(claim_, Q_.partial_me(u1_, seq_len, dim), K_.partial_me(u2_, seq_len, dim), ud_, proof);
        });
    }
    return out__;
}

FrTensor llama_ffn(const FrTensor& X, const LayerWeights& w, const LayerDims& d, tLookupRangeMapping& swiglu, bool prove, LayerTimers& t)
{
    const uint seq_len = d.seq_len, embed_dim = d.embed_dim, hidden_dim = d.hidden_dim;

    zkFC up_layer(embed_dim, hidden_dim, w.up.weight);
    zkFC gate_layer(embed_dim, hidden_dim, w.gate.weight);
    zkFC down_layer(hidden_dim, embed_dim, w.down.weight);

    Rescaling up_rescale(1 << 16);
    Rescaling gate_rescale(1 << 20);
    Rescaling hidden_rescale(1 << 16);
    Rescaling down_rescale(1 << 16);

    FrTensor up_out = timed(t.compute_s, [&] { return up_layer(X); });
    FrTensor up_out_ = timed(t.compute_s, [&] { return up_rescale(up_out); });
    FrTensor gate_out = timed(t.compute_s, [&] { return gate_layer(X); });
    FrTensor gate_out_ = timed(t.compute_s, [&] { return gate_rescale(gate_out); });
    auto p = timed(t.compute_s, [&] { return swiglu(gate_out_); });
    auto &swiglu_out = p.first, &swiglu_m = p.second;
    FrTensor down_in = timed(t.compute_s, [&] { return swiglu_out * up_out_; });
    FrTensor down_in_ = timed(t.compute_s, [&] { return hidden_rescale(down_in); });
    FrTensor down_out = timed(t.compute_s, [&] { return down_layer(down_in_); });
    FrTensor down_out_ = timed(t.compute_s, [&] { return down_rescale(down_out); });

    if (prove) {
        timed(t.prove_s, [&] {
            auto temp_rand = random_vec(3);
            auto swiglu_u = random_vec(ceilLog2(seq_len * hidden_dim));
            auto swiglu_v = random_vec(ceilLog2(seq_len * hidden_dim));
            vector<Polynomial> swiglu_proof;

            down_rescale.prove(down_out, down_out_);
            verifyWeightClaim(w.down, down_layer.prove(down_in_, down_out)[0]);
            hidden_rescale.prove(down_in, down_in_);
            swiglu.prove(gate_out_, swiglu_out, swiglu_m, temp_rand[0], temp_rand[1], temp_rand[2], swiglu_u, swiglu_v, swiglu_proof);
            gate_rescale.prove(gate_out, gate_out_);
            verifyWeightClaim(w.gate, gate_layer.prove(X, gate_out)[0]);
            up_rescale.prove(up_out, up_out_);
            verifyWeightClaim(w.up, up_layer.prove(X, up_out)[0]);
        });
    }
    // Upstream ffn.cu saves the un-rescaled down_out; the rescaled value is the layer's real output.
    return down_out_;
}

FrTensor run_layer(const FrTensor& x_in, const LayerDims& d, const LayerWeights& w, tLookupRangeMapping& swiglu, zkSoftmax& softmax,
    bool prove, LayerTimers& t, BoundaryClaim* input_claim_out)
{
    FrTensor n1 = llama_rmsnorm(x_in, w.input_norm, d, prove, t, input_claim_out);
    FrTensor a = llama_attention(n1, w, d, softmax, prove, t);
    FrTensor h = timed(t.compute_s, [&] { return x_in + a; });
    FrTensor n2 = llama_rmsnorm(h, w.post_norm, d, prove, t, nullptr);
    FrTensor f = llama_ffn(n2, w, d, swiglu, prove, t);
    return timed(t.compute_s, [&] { return h + f; });
}
