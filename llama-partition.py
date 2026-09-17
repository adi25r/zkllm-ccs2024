import os, sys, re, json, time, subprocess, importlib.util
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='LLaMa-2 layer-partitioned proving (zkMod-style)')
parser.add_argument('model_size', type=int, choices=[7, 13], help='The size of the model to use')
parser.add_argument('seq_len', type=int, help='The sequence length to prove (power of two)')
parser.add_argument('--splits', type=int, default=1, help='Number of contiguous layer partitions (even split, remainder to the first ones)')
parser.add_argument('--layers', type=str, default=None, help='Explicit partition sizes, e.g. 8,8,8,8 (overrides --splits)')
parser.add_argument('--num_layers', type=int, default=None, help='Number of decoder layers to prove (default: all)')
parser.add_argument('--devices', type=str, default='0', help='Comma-separated CUDA device ids; partition i runs on devices[i %% n]')
parser.add_argument('--concurrent', type=int, default=None, help='Wave width: partitions proven simultaneously (default: all)')
parser.add_argument('--sequential', action='store_true', help='Prove partitions one at a time (wave width 1)')
parser.add_argument('--input_file', type=str, default='layer_input.bin', help='Input activation (random if missing)')
parser.add_argument('--workdir', type=str, default=None, help='Directory with committed weights (default ./zkllm-workdir/Llama-2-{N}b)')
parser.add_argument('--out-json', type=str, default='partition-summary.json', help='Where to write the JSON summary')
parser.add_argument('--skip-witgen', action='store_true', help='Reuse existing part*_in.bin files from a previous witgen chain')

from transformers import AutoConfig

PARTITION_RE = re.compile(r'^PARTITION (.*)$', re.M)


def parse_partition_line(stdout):
    m = PARTITION_RE.findall(stdout)
    if not m:
        return None
    fields = {}
    for kv in m[-1].split():
        k, v = kv.split('=', 1)
        try:
            fields[k] = float(v) if '.' in v else int(v)
        except ValueError:
            fields[k] = v
    return fields


def ensure_swiglu_table():
    if os.path.isfile('swiglu-table.bin'):
        return
    spec = importlib.util.spec_from_file_location('llama_ffn', 'llama-ffn.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.prepare_swiglu()


def partition_sizes(num_layers, splits, explicit):
    if explicit:
        sizes = [int(s) for s in explicit.split(',')]
        if sum(sizes) != num_layers or any(s <= 0 for s in sizes):
            raise ValueError(f'--layers {explicit} must be positive and sum to {num_layers}')
        return sizes
    base, rem = divmod(num_layers, splits)
    if base == 0:
        raise ValueError(f'--splits {splits} > num_layers {num_layers}')
    return [base + (1 if i < rem else 0) for i in range(splits)]


if __name__ == '__main__':
    args = parser.parse_args()
    if os.system('make partition ppgen'):
        print('Error compiling partition/ppgen')
        exit(1)

    model_card = f'meta-llama/Llama-2-{args.model_size}b-hf'
    config = AutoConfig.from_pretrained(model_card, local_files_only=True, cache_dir='./model-storage')
    embed_dim, hidden_dim = config.hidden_size, config.intermediate_size
    num_layers = args.num_layers or config.num_hidden_layers
    eps = config.rms_norm_eps
    workdir = args.workdir or f'./zkllm-workdir/Llama-2-{args.model_size}b'
    devices = args.devices.split(',')

    ensure_swiglu_table()
    # One shared basis for every boundary commitment. Never regenerate between witgen and prove.
    if not os.path.isfile('act-pp.bin'):
        if os.system(f'./ppgen {embed_dim} act-pp.bin'):
            print('Error generating act-pp.bin')
            exit(1)
    if not os.path.isfile(args.input_file):
        np.round(np.random.randn(args.seq_len, embed_dim) * (1 << 16)).astype(np.int32).tofile(args.input_file)

    sizes = partition_sizes(num_layers, args.splits, args.layers)
    P = len(sizes)
    firsts = [sum(sizes[:i]) for i in range(P)]
    inputs = [args.input_file] + [f'part{i}_in.bin' for i in range(1, P)]
    outputs = [f'part{i + 1}_in.bin' for i in range(P - 1)] + ['final_out.bin']

    def base_cmd(i, mode, output):
        return ['./partition', inputs[i], str(args.seq_len), str(embed_dim), str(hidden_dim), workdir,
                str(firsts[i]), str(sizes[i]), output, mode]

    def env_for(i):
        return {**os.environ, 'CUDA_VISIBLE_DEVICES': devices[i % len(devices)]}

    # Serial witness-generation chain: each partition's input is the previous partition's output.
    prep_s = 0.0
    if not args.skip_witgen:
        t0 = time.time()
        for i in range(P):
            r = subprocess.run(base_cmd(i, 'witgen', outputs[i]), env=env_for(0))
            if r.returncode != 0:
                print(f'witgen of partition {i} failed (exit {r.returncode})')
                exit(1)
        prep_s = time.time() - t0

    # Prove partitions in waves of W concurrent processes, one GPU each.
    W = 1 if args.sequential else (args.concurrent or P)
    results = []
    makespan_s = wall_makespan_s = seq_sum_s = 0.0
    status_ok = True
    for wave_start in range(0, P, W):
        wave = list(range(wave_start, min(wave_start + W, P)))
        t0 = time.time()
        procs = {}
        for i in wave:
            cmd = base_cmd(i, 'prove', f'part{i}_prove_out.bin') + [
                '--part-idx', str(i), '--act-pp', 'act-pp.bin',
                '--commit-input', str(int(i > 0)), '--commit-output', str(int(i < P - 1)),
                '--prefix', f'part{i}', '--eps', repr(eps)]
            procs[i] = subprocess.Popen(cmd, env=env_for(i), stdout=subprocess.PIPE, text=True)
        wave_max = 0.0
        for i in wave:
            out, _ = procs[i].communicate()
            sys.stdout.write(out)
            fields = parse_partition_line(out) or {}
            fields.update(idx=i, exit_code=procs[i].returncode, device=devices[i % len(devices)])
            results.append(fields)
            if procs[i].returncode != 0 or fields.get('status') != 'OK':
                status_ok = False
            else:
                seq_sum_s += fields['prover_total_s']
                wave_max = max(wave_max, fields['prover_total_s'])
        wall_makespan_s += time.time() - t0
        makespan_s += wave_max

    # Cross-partition binding: the producer's output commitment must equal the consumer's input commitment.
    expected_len = args.seq_len * 144
    chain_ok = True
    for i in range(P - 1):
        try:
            a = open(f'part{i}-out-commitment.bin', 'rb').read()
            b = open(f'part{i + 1}-in-commitment.bin', 'rb').read()
            chain_ok &= (a == b) and len(a) == expected_len
        except FileNotFoundError:
            chain_ok = False
    status = 'OK' if status_ok and chain_ok else 'FAIL'

    print(f'SPLIT_SUMMARY parts={P} prep_s={prep_s:.3f} makespan_s={makespan_s:.3f} wall_makespan_s={wall_makespan_s:.3f} '
          f'seq_sum_s={seq_sum_s:.3f} chain_ok={chain_ok} status={status}')
    with open(args.out_json, 'w') as f:
        json.dump({
            'config': {'model_size': args.model_size, 'seq_len': args.seq_len, 'embed_dim': embed_dim, 'hidden_dim': hidden_dim,
                       'num_layers': num_layers, 'sizes': sizes, 'devices': devices, 'wave_width': W, 'eps': eps},
            'partitions': results, 'prep_s': prep_s, 'makespan_s': makespan_s, 'wall_makespan_s': wall_makespan_s,
            'seq_sum_s': seq_sum_s, 'chain_ok': chain_ok, 'status': status,
        }, f, indent=2)
    exit(0 if status == 'OK' else 1)
