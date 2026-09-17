// One partition of a layer-split LLaMA proof: runs decoder layers [first_layer, first_layer+num_layers)
// on an input activation and, in prove mode, commits the boundary tensors and binds the
// partition's first proof to the input commitment.
//
// ./partition <input.bin> <seq_len> <embed_dim> <hidden_dim> <workdir> <first_layer> <num_layers> <output.bin> <witgen|prove>
//             [--part-idx N] [--act-pp act-pp.bin] [--commit-input 0|1] [--commit-output 0|1] [--eps 1e-5]
//             [--prefix partN] [--swiglu-table swiglu-table.bin] [--tamper-after-commit 0|1]

#include "llama-layer.cuh"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

struct Args {
    string input_file, workdir, output_file, mode;
    uint seq_len = 0, embed_dim = 0, hidden_dim = 0, first_layer = 0, num_layers = 0;
    int part_idx = 0;
    string act_pp = "act-pp.bin";
    bool commit_input = false, commit_output = false, tamper_after_commit = false;
    double eps = 1e-5;
    string prefix, swiglu_table = "swiglu-table.bin";
};

[[noreturn]] void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s <input.bin> <seq_len> <embed_dim> <hidden_dim> <workdir> <first_layer> <num_layers> <output.bin> <witgen|prove>"
        " [--part-idx N] [--act-pp file] [--commit-input 0|1] [--commit-output 0|1] [--eps x] [--prefix p]"
        " [--swiglu-table file] [--tamper-after-commit 0|1]\n", argv0);
    exit(2);
}

Args parse_args(int argc, char* argv[])
{
    if (argc < 10) usage(argv[0]);
    Args a;
    a.input_file = argv[1];
    a.seq_len = std::stoul(argv[2]);
    a.embed_dim = std::stoul(argv[3]);
    a.hidden_dim = std::stoul(argv[4]);
    a.workdir = argv[5];
    a.first_layer = std::stoul(argv[6]);
    a.num_layers = std::stoul(argv[7]);
    a.output_file = argv[8];
    a.mode = argv[9];
    if (a.mode != "witgen" && a.mode != "prove") usage(argv[0]);

    for (int i = 10; i + 1 < argc; i += 2) {
        const string key = argv[i], val = argv[i + 1];
        if (key == "--part-idx") a.part_idx = std::stoi(val);
        else if (key == "--act-pp") a.act_pp = val;
        else if (key == "--commit-input") a.commit_input = std::stoi(val) != 0;
        else if (key == "--commit-output") a.commit_output = std::stoi(val) != 0;
        else if (key == "--eps") a.eps = std::stod(val);
        else if (key == "--prefix") a.prefix = val;
        else if (key == "--swiglu-table") a.swiglu_table = val;
        else if (key == "--tamper-after-commit") a.tamper_after_commit = std::stoi(val) != 0;
        else usage(argv[0]);
    }
    if (a.prefix.empty()) a.prefix = "part" + std::to_string(a.part_idx);
    return a;
}

int run(const Args& args)
{
    const bool prove = args.mode == "prove";
    const LayerDims dims{args.seq_len, args.embed_dim, args.hidden_dim, args.eps};

    Timer wall;
    wall.start();

    FrTensor X_in = FrTensor::from_int_bin(args.input_file);
    if (X_in.size != args.seq_len * args.embed_dim)
        throw std::runtime_error("partition: input size " + std::to_string(X_in.size) + " != seq_len*embed_dim");

    // Boundary commitments share one public basis of embed_dim generators, so the tensor is
    // committed as seq_len rows and the producer's and consumer's commitments are identical.
    const bool do_commit_input = prove && args.commit_input;
    const bool do_commit_output = prove && args.commit_output;
    std::unique_ptr<Commitment> act_gen;
    if (do_commit_input || do_commit_output) {
        act_gen = std::make_unique<Commitment>(args.act_pp);
        if (act_gen->size != args.embed_dim)
            throw std::runtime_error("partition: act-pp has " + std::to_string(act_gen->size) + " generators, expected embed_dim");
    }

    Timer t_input_commit, t_output_commit;
    std::unique_ptr<G1TensorJacobian> C_in;
    if (do_commit_input) {
        t_input_commit.start();
        C_in = std::make_unique<G1TensorJacobian>(act_gen->commit_int(X_in));
        t_input_commit.stop();
        C_in->save(args.prefix + "-in-commitment.bin");
        if (args.tamper_after_commit) X_in += {1, 0, 0, 0, 0, 0, 0, 0};
    }

    FrTensor swiglu_values = FrTensor::from_int_bin(args.swiglu_table);
    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);
    zkSoftmax softmax({1 << 8, 1 << 20, 1 << 20}, 1, 0, 1UL << 32, {1 << 18, 1 << 22}, args.seq_len, args.seq_len, args.embed_dim, 1);

    FrTensor x(X_in);
    BoundaryClaim first;
    LayerTimers t;
    for (uint l = args.first_layer; l < args.first_layer + args.num_layers; ++l) {
        LayerWeights w(args.workdir, l, dims);
        x = run_layer(x, dims, w, swiglu, softmax, prove, t, l == args.first_layer ? &first : nullptr);
    }

    if (do_commit_input) {
        t_input_commit.start();
        if (!first.valid) throw std::runtime_error("partition: no input claim produced");
        if (act_gen->open(X_in, *C_in, first.v) != first.value)
            throw std::runtime_error("partition: input boundary opening mismatch");
        t_input_commit.stop();
        cout << "Input boundary opening complete" << endl;
    }

    if (do_commit_output) {
        t_output_commit.start();
        G1TensorJacobian C_out = act_gen->commit_int(x);
        auto r = random_vec(ceilLog2(x.size));
        if (act_gen->open(x, C_out, r) != x(r))
            throw std::runtime_error("partition: output boundary opening mismatch");
        t_output_commit.stop();
        C_out.save(args.prefix + "-out-commitment.bin");
        cout << "Output boundary opening complete" << endl;
    }

    x.save_int(args.output_file);
    wall.stop();

    const double input_commit_s = t_input_commit.getTotalTime();
    const double output_commit_s = t_output_commit.getTotalTime();
    const double prover_total_s = input_commit_s + t.prove_s + output_commit_s;
    printf("PARTITION idx=%d first_layer=%u num_layers=%u witgen_s=%.6f input_commit_s=%.6f prove_s=%.6f output_commit_s=%.6f prover_total_s=%.6f wall_s=%.6f status=OK\n",
        args.part_idx, args.first_layer, args.num_layers, t.compute_s, input_commit_s, t.prove_s, output_commit_s, prover_total_s, wall.getTotalTime());
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    const Args args = parse_args(argc, argv);
    try {
        return run(args);
    } catch (const std::exception& e) {
        fprintf(stderr, "partition: %s\n", e.what());
        printf("PARTITION idx=%d first_layer=%u num_layers=%u status=FAIL\n", args.part_idx, args.first_layer, args.num_layers);
        return 1;
    }
}
