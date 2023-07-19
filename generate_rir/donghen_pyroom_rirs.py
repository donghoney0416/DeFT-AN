import os
import glob
import argparse
import configparser as CP
import multiprocessing
from itertools import repeat

import math
import random
import numpy as np

import pyroomacoustics as pra
import utils
import scipy.io.wavfile


PROCESSES = multiprocessing.cpu_count()


def gen_rirs(params, filenum):
    while True:
        np.random.seed(random.randint(0, 65536))
        max_tries_room = 1000

        # Set RT60
        rt60 = params['room_rt60_min'] + np.random.rand() * (params['room_rt60_max'] - params['room_rt60_min'])
        print(f"Generating RIR #{filenum}, RT60: {rt60}s...")

        # Set room geometry
        width = params['room_width_min'] + np.random.rand() * (params['room_width_max'] - params['room_width_min'])
        length = params['room_length_min'] + np.random.rand() * (params['room_length_max'] - params['room_length_min'])
        height = params['room_height_min'] + np.random.rand() * (params['room_height_max'] - params['room_height_min'])
        room_geometry = np.array([width, length, height])

        # Set microphone array location
        axx = width / 2 + (-0.5) + np.random.rand() * 1
        ayy = length / 2 + (-0.5) + np.random.rand() * 1
        azz = 1 + np.random.rand() * 1

        array_center_location = np.array([axx, ayy, azz])
        mic_angle = np.random.rand() * math.pi / 4

        # Set a speech source location
        cnt = 0
        while True:
            source_location = params['room_offset_inside'] + np.random.rand(3) * (room_geometry - params['room_offset_inside'] * 2)
            src_dist = np.sqrt(np.sum(np.power(source_location - array_center_location, 2)))

            noise_location = params['room_offset_inside'] + np.random.rand(3) * (room_geometry - params['room_offset_inside'] * 2)
            nse_dist = np.sqrt(np.sum(np.power(noise_location - array_center_location, 2)))
            if params['array_source_distance_min'] < src_dist < params['array_source_distance_max']:
                break
            cnt += 1
            if cnt > max_tries_room:
                assert 0, f"Speech source locating failed."

        # Build shoebox room
        # print("Building a shoebox room...")
        # Set absorption coefficients
        e_absorption, _ = pra.inverse_sabine(rt60=rt60, room_dim=room_geometry)
        # reverberant speech
        room1 = pra.ShoeBox(room_geometry, fs=params['fs'], max_order=6, ray_tracing=True, materials=pra.Material(e_absorption))
        room1.set_ray_tracing()
        # direct speech
        room2 = pra.ShoeBox(room_geometry, fs=params['fs'], max_order=0)
        # reverberant noise
        room3 = pra.ShoeBox(room_geometry, fs=params['fs'], max_order=6, ray_tracing=True, materials=pra.Material(e_absorption))
        room3.set_ray_tracing()

        # Add microphone array
        # print("Adding microphone array...")
        # circular array
        mics = pra.beamforming.circular_microphone_array_xyplane(center=array_center_location, M=params['microphone_num'],
                                                                 phi0=mic_angle, radius=params['microphone_radius'],
                                                                 fs=params['fs'], directivity=None)
        # you can set the height of microphone array position and source position same
        # reverberant speech
        room1.add(mics)
        # direct speech
        room2.add(mics)
        # reverberant noise
        room3.add(mics)

        # Add a speech source location
        # print("Adding a speech source location...")
        # reverberant speech
        room1.add_source(position=source_location)
        # direct speech
        room2.add_source(position=source_location)
        # reverberant noise
        room3.add_source(position=noise_location)

        # Compute RIRs
        # print("Computing RIRs...")
        # reverberant speech
        room1.compute_rir()
        # direct speech
        room2.compute_rir()
        # reverberant noise
        room3.compute_rir()

        # filename
        filename1 = f'RIR_{width:.2f}-{length:.2f}-{height:.2f}' \
                    f'_rt60-{room1.rt60_theory():.2f}_index_{filenum}_reverb.wav'
        filename2 = f'RIR_{width:.2f}-{length:.2f}-{height:.2f}' \
                    f'_rt60-{room1.rt60_theory():.2f}_index_{filenum}_direct.wav'
        filename3 = f'RIR_{width:.2f}-{length:.2f}-{height:.2f}' \
                    f'_rt60-{room1.rt60_theory():.2f}_index_{filenum}_noise.wav'
        # pyroomacoustic bugs!
        save_dir1 = os.path.join(params['rir_proc_dir'], filename1)
        save_dir2 = os.path.join(params['rir_proc_dir'], filename2)
        save_dir3 = os.path.join(params['rir_proc_dir'], filename3)

        rirs1 = room1.rir
        rirs2 = room2.rir
        rirs3 = room3.rir

        # To make np.array & zero padding
        rir_len_ls1 = []
        for mic_idx in range(len(rirs1)):
            for src_idx in range(len(rirs1[mic_idx])):
                rir_len_ls1.append(len(rirs1[mic_idx][src_idx]))
        rir_len_max1 = max(rir_len_ls1)
        for mic_idx in range(len(rirs1)):
            for src_idx in range(len(rirs1[mic_idx])):
                rirs1[mic_idx][src_idx] = np.pad(rirs1[mic_idx][src_idx], (0, rir_len_max1 - len(rirs1[mic_idx][src_idx])), 'constant')
        rirs1 = np.array(rirs1)
        if not np.all(np.any(rirs1, axis=-1)):
            continue

        # To make np.array & zero padding
        rir_len_ls2 = []
        for mic_idx in range(len(rirs1)):
            for src_idx in range(len(rirs1[mic_idx])):
                rir_len_ls2.append(len(rirs1[mic_idx][src_idx]))
        rir_len_max2 = max(rir_len_ls2)
        for mic_idx in range(len(rirs2)):
            for src_idx in range(len(rirs2[mic_idx])):
                rirs2[mic_idx][src_idx] = np.pad(rirs2[mic_idx][src_idx], (0, rir_len_max2 - len(rirs2[mic_idx][src_idx])), 'constant')
        rirs2 = np.array(rirs2)
        if not np.all(np.any(rirs2, axis=-1)):
            continue

        # To make np.array & zero padding
        rir_len_ls3 = []
        for mic_idx in range(len(rirs1)):
            for src_idx in range(len(rirs1[mic_idx])):
                rir_len_ls3.append(len(rirs1[mic_idx][src_idx]))
        rir_len_max3 = max(rir_len_ls3)
        for mic_idx in range(len(rirs3)):
            for src_idx in range(len(rirs3[mic_idx])):
                rirs3[mic_idx][src_idx] = np.pad(rirs3[mic_idx][src_idx], (0, rir_len_max3 - len(rirs3[mic_idx][src_idx])), 'constant')
        rirs3 = np.array(rirs3)
        if not np.all(np.any(rirs3, axis=-1)):
            continue

        scipy.io.wavfile.write(save_dir1, params['fs'], np.transpose(np.squeeze(rirs1, axis=1)))
        scipy.io.wavfile.write(save_dir2, params['fs'], np.transpose(np.squeeze(rirs2, axis=1)))
        scipy.io.wavfile.write(save_dir3, params['fs'], np.transpose(np.squeeze(rirs3, axis=1)))

        # try:
        #     np.save(save_dir1, rirs1)
        #     np.save(save_dir2, rirs2)
        #     np.save(save_dir3, rirs3)
        #
        # except Exception as e:
        #     print(str(e))
        #     pass

    return room_geometry


def main_body():
    '''Main body of this file'''
    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='pyroom_rir.cfg',
                        help='Read pyroom_rir.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='pyroom_rir')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    params['fs'] = int(cfg['sampling_rate'])

    params['room_rt60_min'] = float(cfg['room_rt60_min'])
    params['room_rt60_max'] = float(cfg['room_rt60_max'])

    params['room_width_min'] = float(cfg['room_width_min'])
    params['room_width_max'] = float(cfg['room_width_max'])
    params['room_length_min'] = float(cfg['room_length_min'])
    params['room_length_max'] = float(cfg['room_length_max'])
    params['room_height_min'] = float(cfg['room_height_min'])
    params['room_height_max'] = float(cfg['room_height_max'])

    params['room_offset_inside'] = float(cfg['room_offset_inside'])

    params['microphone_num'] = int(cfg['microphone_num'])
    params['microphone_radius'] = float(cfg['microphone_radius'])

    params['array_source_distance_min'] = float(cfg['array_source_distance_min'])
    params['array_source_distance_max'] = float(cfg['array_source_distance_max'])

    if cfg['fileindex_start'] != 'None' and cfg['fileindex_start'] != 'None':
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])
        params['num_files'] = int(params['fileindex_end'])-int(params['fileindex_start'])
    else:
        params['num_files'] = int((params['total_hours']*60*60)/params['audio_length'])

    print('Number of files to be synthesized:', params['num_files'])
    params['is_test_set'] = utils.str2bool(cfg['is_test_set'])
    params['rir_proc_dir'] = utils.get_dir(cfg, 'rir_destination', 'pyroom_RIRs')

    multi_pool = multiprocessing.Pool(processes=PROCESSES)
    fileindices = range(params['num_files'])
    output_lists = multi_pool.starmap(gen_rirs, zip(repeat(params), fileindices))


if __name__ == '__main__':
    main_body()
