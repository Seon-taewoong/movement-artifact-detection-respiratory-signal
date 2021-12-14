## system packages
import xml.etree.ElementTree as ET
import os
## sud-party packages
import pyedflib
import numpy as np
from pandas import DataFrame
from biosppy.signals import tools as st
## custom packages
from libs.signals import smoothing
from libs.signals import make_edr, make_hr, interpolation_1d


def cfs_ecg_load_sleep_stage_slicing(subject_code, abdo_filt=True,
                                     air_flow_binding=False,
                                     ecg_filt_fre_band=[3, 45]):
    # 파일 경로에 따라 바꿀 것
    data_ecg_peak_path = 'F:\ex_data\sleep_org\cfs\polysomnography\cfs_peak'
    cfs_annotation_path = 'F:\ex_data\sleep_org\cfs\polysomnography\\annotations-events-nsrr'
    data_edf_cfs = 'F:\ex_data\sleep_org\cfs\polysomnography\edfs'
    # 고정 값
    subject_name = 'cfs-visit5-%d' % subject_code
    ## start & end info
    tmp_xml_file_name = subject_name + '-nsrr.xml'

    try:
        xml_doc = ET.parse(os.path.join(cfs_annotation_path, tmp_xml_file_name))
    except:
        return 0

    start_sec, end_sec = xml_parsing_start_info(xml_doc)
    try:
        tmp_cfs_edf_f = pyedflib.EdfReader(os.path.join(data_edf_cfs, subject_name + '.edf'))
    except:
        return 0

    ecg_1_num, ecg_2_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ECG1')[0][0], np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ECG2')[0][0]
    spo2_ch_num = np.where(np.logical_or(np.array(tmp_cfs_edf_f.getSignalLabels() )=='SpO2', np.array(tmp_cfs_edf_f.getSignalLabels() )=='SaO2'))[0]
    position_ch_num = np.where(np.logical_or(np.array(tmp_cfs_edf_f.getSignalLabels() )=='POSITION', np.array(tmp_cfs_edf_f.getSignalLabels() )=='Position'))[0][0]
    abdo_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'ABDO EFFORT')[0][0]

    if air_flow_binding:
        air_flow_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'AIRFLOW')[0][0]
        air_pres_num = np.where(np.array(tmp_cfs_edf_f.getSignalLabels()) == 'NASAL PRES')[0][0]
    else:
        air_flow_num = 0
        air_pres_num = 0

    _, ecg_2_fs = tmp_cfs_edf_f.getSampleFrequencies()[ecg_1_num], tmp_cfs_edf_f.getSampleFrequencies()[ecg_2_num]
    abdo_fs = tmp_cfs_edf_f.getSampleFrequencies()[abdo_num]
    position_fs = tmp_cfs_edf_f.getSampleFrequencies()[position_ch_num]

    if air_flow_binding:
        air_flow_fs = tmp_cfs_edf_f.getSampleFrequencies()[air_flow_num]
        air_pres_fs = tmp_cfs_edf_f.getSampleFrequencies()[air_pres_num]
    else:
        air_flow_fs = 0
        air_pres_fs = 0

    ecg = tmp_cfs_edf_f.readSignal(ecg_2_num)[int(start_sec * ecg_2_fs):int(end_sec * ecg_2_fs)]
    ecg_1 = tmp_cfs_edf_f.readSignal(ecg_1_num)[int(start_sec * ecg_2_fs):int(end_sec * ecg_2_fs)]

    if air_flow_binding:
        air_flow = tmp_cfs_edf_f.readSignal(air_flow_num)[int(start_sec * air_flow_fs):int(end_sec * air_flow_fs)]
        air_pres = tmp_cfs_edf_f.readSignal(air_pres_num)[int(start_sec * air_pres_fs):int(end_sec * air_pres_fs)]
    else:
        air_flow = 0
        air_pres = 0

    order = int(0.3 * ecg_2_fs)
    ## ecg filtering
    ecg, _, _ = st.filter_signal(signal=ecg,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=ecg_filt_fre_band,
                                 sampling_rate=ecg_2_fs)

    ecg_sum, _, _ = st.filter_signal(signal=ecg + ecg_1,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order,
                                     frequency=ecg_filt_fre_band,
                                     sampling_rate=ecg_2_fs)
    ## ecg peak
    peak = np.load(os.path.join(data_ecg_peak_path, str(subject_code) + '.npy'))
    ## edf close
    if abdo_filt:
        tmp_abdo = smoothing(tmp_cfs_edf_f.readSignal(abdo_num), 33)
    else:
        tmp_abdo = tmp_cfs_edf_f.readSignal(abdo_num)

    abdo = tmp_abdo[int(start_sec * abdo_fs):int(end_sec * abdo_fs)]
    spo2_raw = tmp_cfs_edf_f.readSignal(spo2_ch_num)[start_sec:end_sec]
    position = tmp_cfs_edf_f.readSignal(position_ch_num)[int(start_sec * position_fs):int(end_sec * position_fs)]
    tmp_cfs_edf_f._close()

    apnea_osa, apnea_central, hypopnea_pre, sp02_de, arousals, sleep_stage, unsure = xml_parsing(xml_doc)
    apnea = (apnea_osa + apnea_central)[int(start_sec * 10):int(end_sec * 10)]
    hypopnea = hypopnea_pre[int(start_sec * 10):int(end_sec * 10)]
    sleep_stage = sleep_stage[int(start_sec * 10):int(end_sec * 10)]
    sp02_de = sp02_de[int(start_sec * 10):int(end_sec * 10)]
    arousals = arousals[int(start_sec * 10):int(end_sec * 10)]
    unsure = unsure[int(start_sec * 10):int(end_sec * 10)]
    edr = smoothing(make_edr(ecg, peak, len(abdo)), 33)
    hr = make_hr(peak, len(abdo))

    return {'signal' :{'ecg' :ecg, 'ecg_peak': peak,
                      'ecg_sum': ecg_sum,
                      'edr' :edr, 'hr': hr, 'abdo': abdo, 'air_flow': air_flow, 'air_pres': air_pres,
                      'spo2' :spo2_raw},
            'label' :{'apnea' :apnea, 'hypopnea' :hypopnea, 'desaturation' :sp02_de, 'arousal' :arousals, 'unsure' :unsure,
                     'sleep_stage': sleep_stage},
            'position' :{'raw_data' :position,
                        'info' :'0: right 1: left 2: back 3: prone '},
            'signal_info' :{'fs_ecg' :ecg_2_fs, 'fs_edr' :abdo_fs, 'fs_abdo' :abdo_fs, 'fs_hr' :abdo_fs, 'fs_air_flow': air_flow_fs, 'fs_air_pres': air_pres_fs,
                           'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10,
                           'fs_position' :1}}


def mesa_load_data_sleep_stage_slicing(subject_code,
                                       filt_abdo=True, label_only=False):
    # 파일 경로에 따라 바꿀것
    mesa_edf_path = 'F:\\ex_data\\sleep_org\\mesa-commercial-use\\polysomnography\\edfs'
    mesa_annotation_path = 'F:\ex_data\sleep_org\mesa-commercial-use\polysomnography\\annotations-events-nsrr'
    mesa_ecg_peak_path = 'F:\ex_data\sleep_org\mesa-commercial-use\polysomnography\ekg_peak\mesa_peak'
    # 고정 값
    mesa_edf_file_name = ('mesa-sleep-%4d.edf' % subject_code).replace(' ', '0')
    mesa_annotation_name = ('mesa-sleep-%4d-nsrr.xml' % subject_code).replace(' ', '0')
    mesa_ecg_peak_name = '%d.npy' % int(subject_code)

    try:
        xml_doc = ET.parse(os.path.join(mesa_annotation_path, mesa_annotation_name))
        tmp_mesa_edf_f = pyedflib.EdfReader(os.path.join(mesa_edf_path, mesa_edf_file_name))
    except:
        return 0
    ## start info
    start_time_sec, end_time_sec, _ = xml_parsing_start_info_flag_ver(xml_doc)

    if end_time_sec == 0:
        return 0

    apnea, apnea_central, hypopnea, sp02_de, arousals, sleep_stage, unsure = xml_parsing(xml_doc)
    ## label info
    mesa_apnea = (apnea + apnea_central)[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_hypopnea = hypopnea[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_sleep_stage = sleep_stage[int(start_time_sec * 10):int(end_time_sec * 10)]
    mesa_unsure = unsure[int(start_time_sec * 10):int(end_time_sec * 10)]
    arousals = arousals[int(start_time_sec * 10):int(end_time_sec * 10)]
    sp02_de = sp02_de[int(start_time_sec * 10):int(end_time_sec * 10)]

    if label_only:
        mesa_spo2 = tmp_mesa_edf_f.readSignal(24)[start_time_sec:end_time_sec]
        mesa_spo2 = np.round(mesa_spo2)
        return {'signal': {'ecg': 0, 'ecg_peak': 0, 'thor': 0,
                           'edr': 0, 'hr': 0, 'abdo': 0,
                           'spo2': mesa_spo2},
                'position': {'raw_data': 0,
                             'info': '0: right 1: back 2: left 3: front 4: upright '},
                'label': {'apnea': mesa_apnea, 'hypopnea': mesa_hypopnea, 'sleep_stage': mesa_sleep_stage,
                          'unsure': mesa_unsure, 'desaturation': sp02_de, 'arousal': arousals},
                'signal_info': {'fs_ecg': 256, 'fs_edr': 32, 'fs_abdo': 32, 'fs_hr': 32, 'fs_thor': 32,
                                'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10, 'fs_unsure': 10}}

    ## signal
    mesa_ekg = -tmp_mesa_edf_f.readSignal(0)[int(start_time_sec * 256):int(end_time_sec * 256)]
    ## define filter order
    order = int(0.3 * 256)
    ## ecg filtering
    mesa_ekg, _, _ = st.filter_signal(signal=mesa_ekg,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=256)
    mesa_ecg_peak = np.load(os.path.join(mesa_ecg_peak_path, mesa_ecg_peak_name), allow_pickle=True)
    if filt_abdo:
        mesa_abdo = smoothing(tmp_mesa_edf_f.readSignal(11)[int(start_time_sec * 32):int(end_time_sec * 32)], 33)
        mesa_thor = smoothing(tmp_mesa_edf_f.readSignal(10)[int(start_time_sec * 32):int(end_time_sec * 32)], 33)
    else:
        mesa_abdo = tmp_mesa_edf_f.readSignal(11)[int(start_time_sec * 32):int(end_time_sec * 32)]
        mesa_thor = tmp_mesa_edf_f.readSignal(10)[int(start_time_sec * 32):int(end_time_sec * 32)]
    mesa_spo2 = tmp_mesa_edf_f.readSignal(24)[start_time_sec:end_time_sec]
    mesa_spo2 = np.round(mesa_spo2)
    mesa_edr = smoothing(make_edr(mesa_ekg, mesa_ecg_peak, len(mesa_abdo)), 33)
    mesa_hr = make_hr(mesa_ecg_peak, len(mesa_abdo))
    mesa_position = tmp_mesa_edf_f.readSignal(14)[int(start_time_sec * 32):int(end_time_sec * 32)]
    tmp_mesa_edf_f._close()

    return {'signal': {'ecg': mesa_ekg, 'ecg_peak': mesa_ecg_peak,
                       'edr': mesa_edr, 'hr': mesa_hr, 'abdo': mesa_abdo, 'thor': mesa_thor,
                       'spo2': mesa_spo2},
            'position': {'raw_data': mesa_position,
                         'info': '0: right 1: back 2: left 3: front 4: upright '},
            'label': {'apnea': mesa_apnea, 'hypopnea': mesa_hypopnea, 'sleep_stage': mesa_sleep_stage,
                      'unsure': mesa_unsure, 'desaturation': sp02_de, 'arousal': arousals},
            'signal_info': {'fs_ecg': 256, 'fs_edr': 32, 'fs_abdo': 32, 'fs_hr': 32, 'fs_thor': 32,
                            'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_sleep_stage': 10, 'fs_unsure': 10}}


def load_kd_data(subject_name, abdo2_smoothing=False):
    ## data path (파일 경로에 따라 바꿀 것)
    KD_DATA_PATH = 'F:\ex_data\sleep_org\kd\\rawdata'
    KD_PEAK_PATH = 'F:\\ex_data\\sleep_org\\kd\\peak'
    KD_PEAK_PATH_ref = 'F:\\ex_data\\sleep_org\\kd\\peak_PSG_ECG_reverse'
    KD_PEAK_PATH_ref_nor_reverse = 'F:\\ex_data\\sleep_org\\kd\\peak_PSG_ECG'
    ## load_data_frame
    kd_data_frame = np.load(os.path.join(KD_DATA_PATH, subject_name+'.npy'), allow_pickle=True).item()
    ## signal binding
    # ecg
    ecg = -kd_data_frame['ECG fs256']
    order = int(0.3 * 256)
    ecg, _, _ = st.filter_signal(signal=ecg,
                                 ftype='FIR',
                                 band='bandpass',
                                 order=order,
                                 frequency=[3, 45],
                                 sampling_rate=256)
    peak = np.load(os.path.join(KD_PEAK_PATH, subject_name + '.npy'))
    # ecg ref (PSG data)
    ecg_ref = kd_data_frame['ECGref fs200']
    order_ref = int(0.3 * 200)
    ecg_ref, _, _ = st.filter_signal(signal=ecg_ref,
                                     ftype='FIR',
                                     band='bandpass',
                                     order=order_ref,
                                     frequency=[3, 45],
                                     sampling_rate=200)
    # ecg peak
    peak_ref = np.load(os.path.join(KD_PEAK_PATH_ref, subject_name + '.npy'))
    peak_ref_reverse = np.load(os.path.join(KD_PEAK_PATH_ref_nor_reverse, subject_name + '.npy'))
    # abdo 1
    abdo1 = smoothing(kd_data_frame['ACC_X fs32'], 33)
    # abdo 2
    if abdo2_smoothing:
        abdo2 = smoothing(kd_data_frame['abdo fs200'], 201)
    else:
        abdo2 = kd_data_frame['abdo fs200']
    # acc signal
    acc_x = kd_data_frame['ACC_X fs32']
    acc_y = kd_data_frame['ACC_Y fs32']
    acc_z = kd_data_frame['ACC_Z fs32']
    # hr
    hr = make_hr(peak, len(acc_x))
    hr_ref = make_hr(peak_ref, len(acc_x))
    # edr
    edr = smoothing(make_edr(ecg, peak, len(acc_x)))
    edr_ref = smoothing(make_edr(ecg_ref, peak_ref, len(acc_x)))
    # spo2
    spo2 = kd_data_frame['spo2 fs200']
    spo2 = interpolation_1d(spo2, int(len(spo2)/200))
    ## label binding
    apnea = kd_data_frame['apnea fs10']
    hypopnea = kd_data_frame['hypopnea fs10']
    flow_limitation = kd_data_frame['flowlimitation fs10']
    arousal = kd_data_frame['wake fs1']
    sleep_stage = kd_data_frame['sstage fs1']

    return {'signal':{'ecg': ecg, 'ecg_ref': ecg_ref, 'ecg_peak': peak, 'ecg_peak_ref': peak_ref, 'ecg_peak_ref_reverse': peak_ref_reverse,
                      'abdo_ref': abdo2,
                      'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
                      'hr': hr, 'hr_ref': hr_ref,
                      'edr': edr, 'edr_ref': edr_ref,
                      'spo2': spo2},
            'label':{'apnea': apnea, 'hypopnea': hypopnea, 'flow_limitation': flow_limitation,
                     'arousal': arousal, 'sleep_stage': sleep_stage},
            'signal_info':{'fs_ecg': 256,
                           'fs_abdo': 32, 'fs_abdo2': 32,
                           'fs_hr': 32, 'fs_edr': 32,
                           'fs_spo2': 1,
                           'fs_apnea': 10, 'fs_hypopnea': 10, 'fs_flowlimitation': 10, 'fs_arousal': 1,
                           'fs_sleep_stage': 1}}


def xml_parsing_start_info(xml_doc):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        tmp_event_type = tmp_info_xml.findtext('EventType')
        if 'Stages' in tmp_event_type:
            event_type.append(tmp_info_xml.findtext('EventType'))
            event_concept.append(tmp_info_xml.findtext('EventConcept'))
            start.append(tmp_info_xml.findtext('Start'))
            duration.append(tmp_info_xml.findtext('Duration'))
            spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
            spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))
        else:
            pass

    tmp_first_stage_start_time = np.float32(start[0])
    tmp_first_stage_duration = np.float32(duration[0])
    tmp_first_concep = event_concept[0]

    if np.logical_or('Wake' in tmp_first_concep, 'Unscored' in tmp_first_concep):
        tmp_start_sec = int(tmp_first_stage_duration - tmp_first_stage_start_time)
    else:
        tmp_start_sec = int(tmp_first_stage_start_time)

    tmp_end_start_time = np.float32(start[-1])
    tmp_end_duration = np.float32(duration[-1])
    tmp_end_concep = event_concept[-1]
    tmp_end_concep_pre = event_concept[-2]
    tmp_end_start_time_pre = np.float32(start[-2])
    if np.logical_or('Wake' in tmp_end_concep, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time)
    else:
        tmp_end_sec = int(tmp_end_start_time + tmp_end_duration)

    if np.logical_and('Wake' in tmp_end_concep_pre, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time_pre)
    else:
        pass

    return tmp_start_sec, tmp_end_sec


def xml_parsing(xml_doc, spo2_desa_index=3):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        event_type.append(tmp_info_xml.findtext('EventType'))
        event_concept.append(tmp_info_xml.findtext('EventConcept'))
        start.append(tmp_info_xml.findtext('Start'))
        duration.append(tmp_info_xml.findtext('Duration'))
        spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
        spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))

    data_len_sec = int(np.float32(duration[0]))
    db = DataFrame({'evnet_type': event_type[1::],
                    'event_concept': event_concept[1::],
                    'start_sec': start[1::],
                    'duration_sec': duration[1::],
                    'spo2_base': spo2_base[1::],
                    'spo2_nar': spo2_nar[1::], })

    apnea, sp02_de, apnea_central, hypopnea, arousals, sleep_stage = np.zeros([data_len_sec * 10]), np.zeros(
        [data_len_sec * 10]), np.zeros([data_len_sec * 10]), \
                                                                     np.zeros([data_len_sec * 10]), np.zeros(
        [data_len_sec * 10]), np.ones([data_len_sec * 10]) * 10
    unsure = np.zeros([data_len_sec * 10])

    for tmp_db in db.iterrows():
        tmp_db = tmp_db[1]
        tmp_event_concept = tmp_db['event_concept']
        tmp_event_type = tmp_db['evnet_type']
        tmp_start_sec = tmp_db['start_sec']
        tmp_duration = tmp_db['duration_sec']
        tmp_spo2_base = tmp_db['spo2_base']
        tmp_spo2_nar = tmp_db['spo2_nar']

        start_cursor = int(np.float32(tmp_start_sec) * 10)
        end_cursor = start_cursor + int(np.float32(tmp_duration) * 10)

        if 'Hypopnea' in tmp_event_concept:
            hypopnea[start_cursor:end_cursor] = 1
        elif 'SpO2 desaturation' in tmp_event_concept:
            if (np.float32(tmp_spo2_base) - np.float32(tmp_spo2_nar)) >= spo2_desa_index:
                sp02_de[start_cursor:end_cursor] = 1
        elif 'Central apnea' in tmp_event_concept:
            apnea_central[start_cursor:end_cursor] = 1
        elif 'Obstructive apnea' in tmp_event_concept:
            apnea[start_cursor:end_cursor] = 1
        elif tmp_event_concept == 'REM sleep|5':
            sleep_stage[start_cursor:end_cursor] = 4
        elif tmp_event_concept == 'Stage 1 sleep|1':
            sleep_stage[start_cursor:end_cursor] = 1
        elif tmp_event_concept == 'Stage 2 sleep|2':
            sleep_stage[start_cursor:end_cursor] = 2
        elif tmp_event_concept == 'Stage 3 sleep|3':
            sleep_stage[start_cursor:end_cursor] = 3
        elif tmp_event_concept == 'Wake|0':
            sleep_stage[start_cursor:end_cursor] = 0
        elif 'Arousal' in tmp_event_concept:
            arousals[start_cursor:end_cursor] = 1
        elif np.logical_and('Respiratory' in tmp_event_type,
                            'Unsure' in tmp_event_concept):
            unsure[start_cursor:end_cursor] = 1

    return apnea, apnea_central, hypopnea, sp02_de, arousals, sleep_stage, unsure


def xml_parsing_start_info_flag_ver(xml_doc):
    root = xml_doc.getroot()
    event_type, event_concept, start, duration, spo2_base, spo2_nar = [], [], [], [], [], []

    for tmp_info_xml in root.iter('ScoredEvent'):
        tmp_event_type = tmp_info_xml.findtext('EventType')
        if 'Stages' in tmp_event_type:
            event_type.append(tmp_info_xml.findtext('EventType'))
            event_concept.append(tmp_info_xml.findtext('EventConcept'))
            start.append(tmp_info_xml.findtext('Start'))
            duration.append(tmp_info_xml.findtext('Duration'))
            spo2_base.append(tmp_info_xml.findtext('SpO2Baseline'))
            spo2_nar.append(tmp_info_xml.findtext('SpO2Nadir'))
        else:
            pass

    tmp_first_stage_start_time = np.float32(start[0])
    tmp_first_stage_duration = np.float32(duration[0])
    tmp_first_concep = event_concept[0]

    if np.logical_or('Wake' in tmp_first_concep, 'Unscored' in tmp_first_concep):
        tmp_start_sec = int(tmp_first_stage_duration - tmp_first_stage_start_time)
    else:
        tmp_start_sec = int(tmp_first_stage_start_time)

    tmp_end_start_time = np.float32(start[-1])
    tmp_end_duration = np.float32(duration[-1])
    tmp_end_concep = event_concept[-1]
    tmp_end_concep_pre = event_concept[-2]
    tmp_end_start_time_pre = np.float32(start[-2])
    if np.logical_or('Wake' in tmp_end_concep, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time)
        flag = 0
    else:
        tmp_end_sec = int(tmp_end_start_time + tmp_end_duration)
        flag = 0

    if np.logical_and('Wake' in tmp_end_concep_pre, 'Unscored' in tmp_end_concep):
        tmp_end_sec = int(tmp_end_start_time_pre)
        flag = 1
    else:
        pass

    return tmp_start_sec, tmp_end_sec, flag
