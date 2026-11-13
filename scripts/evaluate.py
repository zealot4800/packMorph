"""Evaluate baseline or supported packet-defense features in test.pkl files."""
from __future__ import annotations

import argparse
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from packmorph.baseline import load_baseline
from packmorph.blanket import Blanket
from packmorph.config import add_dataset_arguments, dataset_config_path, METHODS, load_config, policy_parameters
from packmorph.ditto import Ditto
from packmorph.flowshift import FlowShift
from packmorph.features import load_pickle_splits
from packmorph.metrics import evaluate
from packmorph.minipatch import Minipatch
from packmorph.minos import Minos
from packmorph.securitas import Securitas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('strategy', choices=('baseline', *METHODS, 'all'))
    add_dataset_arguments(parser)
    args = parser.parse_args()
    try:
        config = load_config(dataset_config_path(args))
        methods = (() if args.strategy == 'baseline' else
                   tuple(config['obfuscation']) if args.strategy == 'all' else (args.strategy,))
        if any(method not in config['obfuscation'] for method in methods):
            raise ValueError('The requested strategy is missing from obfuscation settings')
        for environment in config['data']['environments']:
            settings = {method: policy_parameters(config, environment, method) for method in methods}
            splits = load_pickle_splits(config, environment)
            frozen = load_baseline(config, environment, splits)
            test = splits['test']
            with (Path(config['data']['root']) / environment / 'test.pkl').open('rb') as handle:
                policy_tests = pickle.load(handle).get('obfuscations', {})
            artifacts = {}
            if any(m in methods for m in ('flowshift', 'blanket', 'minipatch', 'ditto', 'securitas')):
                directory = Path(config['data']['root']) / environment
                with (directory / 'train.pkl').open('rb') as handle:
                    training_saved = pickle.load(handle)
                with (directory / 'val.pkl').open('rb') as handle:
                    validation_saved = pickle.load(handle)
                if 'flowshift' in methods:
                    artifact = training_saved.get('policies', {}).get('flowshift')
                    signature = FlowShift.signature(splits['train'], training_saved.get('flow_context'),
                        settings['flowshift'], config['model']['split_seed'], frozen,
                        splits['validation'], validation_saved.get('flow_context'))
                    FlowShift.verify(artifact, signature)
                    artifacts['flowshift'] = artifact
                for method, cls in (('blanket', Blanket), ('minipatch', Minipatch)):
                    if method not in methods:
                        continue
                    training, train_rows = cls._prepared(splits['train'], training_saved.get('flow_context'))
                    validation, val_rows = cls._prepared(splits['validation'], validation_saved.get('flow_context'))
                    signature = cls.signature(training, train_rows, validation, val_rows, frozen,
                        settings[method], config['model']['split_seed'])
                    artifact = training_saved.get('policies', {}).get(method)
                    if method == 'blanket':
                        cls.verify(artifact, signature)
                    else:
                        cls.verify(artifact, signature, training.class_names, settings[method])
                    artifacts[method] = artifact
                if 'ditto' in methods:
                    training, rows = Ditto._prepared(splits['train'], training_saved.get('flow_context'))
                    signature = Ditto.signature(training, rows, settings['ditto'], config['model']['split_seed'])
                    artifact = training_saved.get('policies', {}).get('ditto')
                    Ditto.verify(artifact, signature)
                    artifacts['ditto'] = artifact
                if 'securitas' in methods:
                    seed = config['model']['split_seed']
                    training, tr = Securitas._prepared(splits['train'], training_saved.get('flow_context'), seed)
                    validation, vr = Securitas._prepared(splits['validation'], validation_saved.get('flow_context'), seed)
                    signature = Securitas.signature(training, tr, validation, vr, settings['securitas'], seed)
                    artifact = training_saved.get('policies', {}).get('securitas')
                    Securitas.verify(artifact, signature)
                    artifacts['securitas'] = artifact

            def score(data):
                return evaluate(frozen.model, frozen.scaler.transform(data.features),
                                data.labels, data.captures, data.class_names,
                                config['model']['batch_size'])

            baseline = score(test)
            print(f'\n{environment}: {len(test)} test.pkl flows; {len(test.feature_names)} model features')
            if not test.capture_disjoint_verified:
                print('  Flow-level holdout; capture separation is unverified.')
            print(f'{"Policy":<12} {"Accuracy":>10} {"Macro F1":>10} {"Accuracy drop":>14}')
            print(f'{"original":<12} {baseline["accuracy"]:>9.2%} {baseline["macro_f1"]:>9.2%} {0:>11.2f} pp', flush=True)
            for method in methods:
                cached = policy_tests.get(method, {})
                if cached.get('parameters') != settings[method]:
                    raise ValueError(f'{method}: test.pkl needs policy features matching the current settings')
                if method in artifacts and (cached.get('generator') != artifacts[method]['version']
                        or cached.get('policy_fingerprint') != artifacts[method]['fingerprint']
                        or cached.get('seed') != config['model']['split_seed']):
                    raise ValueError(f'{method} features are stale; run packmorph-obfuscate {method}')
                if method == 'minos' and (cached.get('generator') != Minos.VERSION
                        or cached.get('seed') != config['model']['split_seed']):
                    raise ValueError('Minos features are stale; run packmorph-obfuscate minos')
                features = np.asarray(cached.get('X'))
                if (features.shape != test.features.shape or features.dtype != np.float32
                        or not np.isfinite(features).all() or np.any(features < 0)):
                    raise ValueError(f'{method}: invalid policy features in test.pkl')
                metrics = score(replace(test, features=features))
                drop = 100 * (baseline['accuracy'] - metrics['accuracy'])
                print(f'{method:<12} {metrics["accuracy"]:>9.2%} {metrics["macro_f1"]:>9.2%} {drop:>11.2f} pp', flush=True)
                if len(methods) == 1:
                    print(f'{"Service":<15} {"Flows":>6} {"Orig recall":>12} {"Recall":>10} {"F1":>10}')
                    for before, after in zip(baseline['per_class'], metrics['per_class']):
                        print(f'{after["name"]:<15} {after["support"]:>6} {before["recall"]:>11.2%}'
                              f' {after["recall"]:>9.2%} {after["f1"]:>9.2%}', flush=True)
            if 'flowshift' in methods:
                overhead = policy_tests['flowshift']['overhead']
                print(f'FlowShift: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_added_span_s"]:.6f}s mean modeled span; '
                      f'{overhead["max_extra_byte_fraction"]:.2%} maximum per-flow extra bytes')
            if 'blanket' in methods:
                overhead = policy_tests['blanket']['overhead']
                print(f'Blanket: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_added_span_s"]:+.6f}s mean modeled span; packet-derived bulk statistics')
            if 'minos' in methods:
                overhead = policy_tests['minos']['overhead']
                print(f'Minos: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_dummy_packets"]:.2f} mean dummies; single-flow morphing, no tunnel mixing')
            if 'minipatch' in methods:
                overhead = policy_tests['minipatch']['overhead']
                print(f'MiniPatch: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_dummy_packets"]:.2f} mean dummies; known-service burst patches')
            if 'ditto' in methods:
                overhead, artifact = policy_tests['ditto']['overhead'], artifacts['ditto']
                print(f'Ditto: pattern={artifact["pattern"]}; {1e9 / artifact["slot_ns"]:.2f} pps/direction; '
                      f'{overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_completion_delay_s"]:.6f}s mean modeled completion delay; isolated carrier windows')
            if 'securitas' in methods:
                overhead = policy_tests['securitas']['overhead']
                print(f'Securitas: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                      f'{overhead["mean_fragmented_packets"]:.2f} fragments, {overhead["mean_fake_packets"]:.2f} fakes, '
                      f'{overhead["mean_icmp_replies"]:.2f} modeled ICMP replies per flow; independent proxy')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted\n')
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Evaluation stopped: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
