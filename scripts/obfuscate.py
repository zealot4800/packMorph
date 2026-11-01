"""Generate fresh policy features in PKLs that contain supporting flow_context."""
from __future__ import annotations

import argparse
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np

from packmorph.baseline import load_baseline
from packmorph.blanket import Blanket
from packmorph.config import add_dataset_arguments, dataset_config_path, METHODS, load_config, policy_parameters
from packmorph.ditto import Ditto
from packmorph.flowshift import FlowShift
from packmorph.features import load_pickle_splits, require_packet_dataset
from packmorph.minipatch import Minipatch
from packmorph.securitas import Securitas
from packmorph.transform import transform_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('strategy', choices=(*METHODS, 'all'))
    add_dataset_arguments(parser)
    parser.add_argument('--environment', choices=('all', 'vpn', 'non-vpn'), default='all')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    parser.add_argument('--all-flows', action='store_true', help='transform all three balanced PKL splits')
    parser.add_argument('--refit', action='store_true', help='refit learned policies')
    parser.add_argument('--flowshift-policy', type=Path, help='use a verified frozen FlowShift study artifact')
    args = parser.parse_args()
    if args.flowshift_policy and (args.strategy != 'flowshift' or args.environment == 'all' or args.refit):
        parser.error('--flowshift-policy requires flowshift, one --environment, and no --refit')
    args.config = dataset_config_path(args)
    temporary_files = []
    try:
        config = load_config(args.config, dataset=args.dataset)
        require_packet_dataset(config)
        environments = (config['data']['environments'] if args.environment == 'all' else
                        ('VPN',) if args.environment == 'vpn' else ('Non VPN',))
        methods = METHODS if args.strategy == 'all' else (args.strategy,)
        filenames = ('train', 'val', 'test') if args.all_flows else (args.split,)
        pending = {}
        summaries = []
        # Validate and compute everything before changing any input file.
        for environment in environments:
            settings = {method: policy_parameters(config, environment, method) for method in methods}
            directory = Path(config['data']['root']) / environment
            splits = load_pickle_splits(config, environment)
            with (directory / 'train.pkl').open('rb') as handle:
                training_saved = pickle.load(handle)
            training_context = training_saved.get('flow_context')
            learners, artifacts = {}, {}
            if any(m in methods for m in ('flowshift', 'securitas', 'blanket', 'minipatch')):
                with (directory / 'val.pkl').open('rb') as handle:
                    validation_context = pickle.load(handle).get('flow_context')
            if 'flowshift' in methods:
                cached = training_saved.get('policies', {}).get('flowshift')
                teacher = (load_baseline(config, environment, splits)
                           if settings['flowshift'].get('selection') == 'distilled' else None)
                if args.flowshift_policy:
                    with args.flowshift_policy.open('rb') as handle:
                        cached = pickle.load(handle)
                    FlowShift.verify(cached, FlowShift.signature(splits['train'], training_context,
                        settings['flowshift'], config['model']['split_seed'], teacher,
                        splits['validation'], validation_context))
                artifact = FlowShift.fit(splits['train'], training_context, settings['flowshift'],
                    config['model']['split_seed'], cached=None if args.refit else cached, frozen=teacher,
                    validation=splits['validation'], validation_context=validation_context)
                if artifact is not training_saved.get('policies', {}).get('flowshift'):
                    training_saved.setdefault('policies', {})['flowshift'] = artifact
                    pending[directory / 'train.pkl'] = training_saved
                learners['flowshift'] = FlowShift(settings['flowshift'], artifact,
                    splits['train'].feature_names, config['model']['split_seed'])
                artifacts['flowshift'] = artifact
            if 'securitas' in methods:
                cached = training_saved.get('policies', {}).get('securitas')
                artifact = Securitas.fit(splits['train'], training_context, splits['validation'],
                    validation_context, settings['securitas'], config['model']['split_seed'],
                    cached=None if args.refit else cached, device=config['model']['device'])
                if artifact is not cached:
                    training_saved.setdefault('policies', {})['securitas'] = artifact
                    pending[directory / 'train.pkl'] = training_saved
                learners['securitas'] = Securitas(settings['securitas'], artifact,
                    splits['train'].feature_names, config['model']['split_seed'])
                artifacts['securitas'] = artifact
            if 'ditto' in methods:
                cached = training_saved.get('policies', {}).get('ditto')
                artifact = Ditto.fit(splits['train'], training_context, settings['ditto'],
                    config['model']['split_seed'], cached=None if args.refit else cached)
                if artifact is not cached:
                    training_saved.setdefault('policies', {})['ditto'] = artifact
                    pending[directory / 'train.pkl'] = training_saved
                learners['ditto'] = Ditto(settings['ditto'], artifact,
                    splits['train'].feature_names, config['model']['split_seed'])
                artifacts['ditto'] = artifact
            if any(m in methods for m in ('blanket', 'minipatch')):
                frozen = load_baseline(config, environment, splits)
                for method, cls in (('blanket', Blanket), ('minipatch', Minipatch)):
                    if method not in methods:
                        continue
                    cached = training_saved.get('policies', {}).get(method)
                    artifact = cls.fit(splits['train'], training_context, splits['validation'],
                        validation_context, frozen, settings[method],
                        config['model']['split_seed'], cached=None if args.refit else cached)
                    if artifact is not cached:
                        training_saved.setdefault('policies', {})[method] = artifact
                        pending[directory / 'train.pkl'] = training_saved
                    controls = artifact['controls'] if method == 'blanket' else artifact
                    learners[method] = cls(settings[method], controls,
                        splits['train'].feature_names, config['model']['split_seed'])
                    artifacts[method] = artifact
            for filename in filenames:
                split = 'validation' if filename == 'val' else filename
                data = splits[split]
                path = directory / f'{filename}.pkl'
                if filename == 'train':
                    saved = training_saved
                else:
                    with path.open('rb') as handle:
                        saved = pickle.load(handle)
                policies = dict(saved.get('obfuscations', {}))
                for method in methods:
                    parameters = settings[method]
                    summary = {}
                    try:
                        started = time.perf_counter()
                        features = transform_dataset(
                            data, method, parameters, context=saved.get('flow_context'),
                            seed=config['model']['split_seed'], fitted_policy=learners.get(method), summary=summary)
                        if method == 'flowshift':
                            summary['generation_seconds'] = time.perf_counter() - started
                    except (ValueError, TypeError, KeyError) as error:
                        raise ValueError(f'{environment}/{filename}.pkl, {method}: {error}') from error
                    policies[method] = dict(X=features, parameters=parameters,
                                            generator='aggregate-flow-v1', seed=config['model']['split_seed'])
                    policies[method].update(summary)
                    if method in learners:
                        artifact, learner = artifacts[method], learners[method]
                        policies[method].update(generator=learner.VERSION,
                            policy_fingerprint=artifact['fingerprint'], overhead=learner.overhead,
                            assumptions=artifact['assumptions'])
                    changed = int(np.any(features != data.features, axis=1).sum())
                    summaries.append(f'{environment} | {filename} | {method}: {changed}/{len(data)} flows changed; {len(data.feature_names)} features')
                    if method == 'flowshift':
                        overhead = learners[method].overhead
                        summaries.append(f'  FlowShift: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_added_span_s"]:.6f}s mean modeled span; '
                            f'{summary["diagnostics"].get("fallback_flows", 0)} identity fallbacks')
                    elif method == 'blanket':
                        overhead = learner.overhead
                        summaries.append(f'  Blanket: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_added_span_s"]:+.6f}s mean modeled span; packet-derived bulk statistics')
                    elif method == 'minos':
                        overhead = summary['overhead']
                        summaries.append(f'  Minos: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_dummy_packets"]:.2f} mean dummies; single-flow morphing, no tunnel mixing')
                    elif method == 'minipatch':
                        overhead = learners[method].overhead
                        summaries.append(f'  MiniPatch: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_dummy_packets"]:.2f} mean dummies; known-service burst patches')
                    elif method == 'ditto':
                        overhead = learners[method].overhead
                        summaries.append(f'  Ditto: pattern={artifact["pattern"]}; {1e9 / artifact["slot_ns"]:.2f} pps/direction; '
                            f'{overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_completion_delay_s"]:.6f}s mean modeled completion delay; isolated carrier windows')
                    elif method == 'securitas':
                        overhead = learners[method].overhead
                        summaries.append(f'  Securitas: {overhead["extra_byte_fraction"]:+.2%} bytes; '
                            f'{overhead["mean_fragmented_packets"]:.2f} fragments, {overhead["mean_fake_packets"]:.2f} fakes, '
                            f'{overhead["mean_icmp_replies"]:.2f} modeled ICMP replies per flow; independent proxy')
                saved['obfuscations'] = policies
                pending[path] = saved
        for path, saved in pending.items():
            with tempfile.NamedTemporaryFile('wb', dir=path.parent, prefix=f'.{path.stem}-',
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                temporary_files.append((temporary, path))
                pickle.dump(saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.chmod(temporary, path.stat().st_mode & 0o777)
        for temporary, path in temporary_files:
            os.replace(temporary, path)
        for summary in summaries:
            print(summary, flush=True)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted\n')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.exit(1, f'Obfuscation stopped: {error}\n')
    finally:
        for temporary, _ in temporary_files:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
