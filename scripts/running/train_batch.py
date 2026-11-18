# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Train independent Mip-NeRF 360 scenes on one device. Pass native training flags after --."""
import argparse
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

PROCESS_STARTED = perf_counter()

from eval_sweep import BASE_ARGS, DATASET_ROOTS, DEFAULT_SCENES, scene_downsample
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import parse_args
from trainer.frontend.reporting.error_log import ErrorLog


class TimedApplication(TrainingApplication):
    def prepare_training_context(self):
        started = perf_counter()
        super().prepare_training_context()
        self.device.wait_for_idle()
        self.setup_seconds = perf_counter() - started


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', default='all', help='Comma-separated scene names, or all nine')
    parser.add_argument('--datasets', type=Path, default=Path('datasets'), help='Directory containing the scene families')
    parser.add_argument('--prefix', required=True, help='New run prefix under outputs/train')
    split = argv.index('--') if '--' in argv else len(argv)
    options = parser.parse_args(argv[:split])
    names = DEFAULT_SCENES if options.scenes == 'all' else options.scenes.split(',')
    jobs = []
    if len(names) != len(set(names)):
        parser.error('Each scene must appear once per batch')
    for name in names:
        locations = (options.datasets / family / name for family in DATASET_ROOTS)
        scene = next((location for location in locations if location.is_dir()), None)
        if scene is None:
            parser.error(f'Unknown scene {name}')
        args = parse_args(['--iters', '3000', *BASE_ARGS, '--eval-every', '0', '--preview-every', '0',
                           '--no-write-ground-truths', *argv[split+1:], '--scene', str(scene),
                           '--downsample', str(scene_downsample(name)), '--run-name', f'{options.prefix}_{name}'])
        if not args.no_viewer or args.hub or args.profile_run or not (args.evaluation and args.final_test_psnr):
            parser.error('Batch mode requires headless training with evaluation and final test PSNR enabled, and profiling off')
        if (Path('outputs/train') / args.run_name).exists():
            parser.error(f'Output already exists: {args.run_name}')
        jobs.append((name, args))
    sessions = None
    for name, args in jobs:
        cold = sessions is None
        started = perf_counter()
        app = TimedApplication(args, slang_sessions=sessions)
        with ErrorLog(app.paths.root / 'errors.log') as errors:
            app.error_log = errors
            app.start_training()
        app.device.wait_for_idle()
        sessions = app.slang_sessions
        metrics = json.loads((app.paths.root / 'test_psnr.json').read_text())
        row = dict(scene=name, cold_device=cold, setup_s=app.setup_seconds, train_loop_s=metrics['train_wall_s'],
                   setup_plus_train_s=app.setup_seconds + metrics['train_wall_s'], psnr_db=metrics['psnr_srgb'])
        peak, overflows = app.pipeline.buffers.sort.stats.to_numpy().view('uint32')
        row.update(peak_sort_pairs_through_evaluation=int(peak), sort_overflows_through_evaluation=int(overflows))
        row['process_startup_s'] = started - PROCESS_STARTED if cold else 0.
        destination = app.paths.root / 'batch_timing.json'
        del app
        gc.collect()
        sessions.device.wait_for_idle()
        row['total_including_eval_s'] = perf_counter() - started
        destination.write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        print('[batch] ' + json.dumps(row), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
