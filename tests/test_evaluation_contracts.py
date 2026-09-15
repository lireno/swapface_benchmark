from pathlib import Path
import ast
import json
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from swapface_benchmark.roi import read_face_boxes, scaled_box
from swapface_benchmark.metrics import identity_strict as strict
from tools.evaluation_fingerprint import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def boxes_file(tmp_path, data):
    p = tmp_path/'boxes.json'
    p.write_text(json.dumps(data))
    return p


def test_boxes_keep_numeric_order_without_interpolation(tmp_path):
    p = boxes_file(tmp_path, {'102': [3,4,7,9], '100': [1,2,5,6]})
    boxes = read_face_boxes(p)
    assert boxes == [(1,2,5,6), (3,4,7,9)]
    assert scaled_box(boxes[1], (10,10), (20,30)) == (9,8,21,18)


@pytest.mark.parametrize('box', [None, [1,2,3], [2,2,1,3], [0,0,float('nan'),4]])
def test_invalid_box_is_rejected_not_dropped(tmp_path, box):
    with pytest.raises(ValueError):
        read_face_boxes(boxes_file(tmp_path, {'1': [0,0,2,2], '2': box}))


def test_no_neighbour_box_fallback(tmp_path):
    (tmp_path/'face_boxes.json').write_text('{"0":[0,0,1,1]}')
    mask = tmp_path/'mask.mp4'
    assert strict.resolve_face_boxes_path({'ref_video_facemask': str(mask)}, tmp_path/'origin.mp4', mask) is None
    with pytest.raises(ValueError, match='missing face box'):
        strict.bbox_from_face_boxes({0:(0,0,1,1)}, 1, (2,2), (2,2))


def write_video(path, count=4):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (32,32))
    assert w.isOpened()
    for i in range(count): w.write(np.full((32,32,3), i*30+40, np.uint8))
    w.release()


def test_reader_handles_duplicate_indices_and_rejects_short_decode(tmp_path):
    p = tmp_path/'video.mp4';write_video(p)
    frames = strict.read_frames(p, [0,0,2,3])
    assert len(frames) == 4
    np.testing.assert_array_equal(frames[0], frames[1])
    assert frames[-1].mean() > frames[0].mean()
    with pytest.raises(RuntimeError, match='incomplete decode'):
        strict.read_frames(p,[0,9])


def test_strict_identity_never_reads_gan(tmp_path, monkeypatch):
    p=tmp_path/'video.mp4';write_video(p)
    ref=tmp_path/'ref.png';cv2.imwrite(str(ref),np.full((32,32,3),100,np.uint8))
    b=boxes_file(tmp_path,{str(i):[8,8,24,24] for i in range(4)})
    seen=[];original=strict.read_frames
    def read(path, indices):
        seen.append(path);return original(path,indices)
    monkeypatch.setattr(strict,'read_frames',read)
    model=SimpleNamespace(embed=lambda frame:np.array([1.,0.],np.float32))
    row={'name':'case','facebench_video_id':'00001','generated':str(p),'ref_video':str(p),
         'ref_image':str(ref),'ref_video_facemask':str(b),'ground_truth':'/nonexistent/GAN.mp4'}
    out=strict.evaluate_case(model,row,4,4,42,False,'face-box',1)
    assert out['id_sim']==1 and out['input_leak']==1
    assert len(seen)==2 and out['gt_ref'] is None


def test_prepare_results_accepts_missing_gan_and_layout_writes_no_video(tmp_path):
    from tools import prepare_results, prepare_facebench_layout
    p=tmp_path/'case.mp4';p.touch();ref=tmp_path/'ref.png';ref.touch()
    b=boxes_file(tmp_path,{'0':[0,0,2,2]})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':'case','origin_video':str(p),
        'ref_image':str(ref),'face_boxes':str(b),'gan_swapped_video':'missing-gan.mp4'}]}))
    mapping=tmp_path/'mapping.json'
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_results.py'),'--manifest',str(manifest),
        '--results-dir',str(tmp_path),'--output',str(mapping)],check=True,capture_output=True)
    row=json.loads(mapping.read_text())[0];assert 'ground_truth' not in row
    layout=tmp_path/'layout'
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_facebench_layout.py'),'--mapping',str(mapping),
        '--output-root',str(layout)],check=True,capture_output=True)
    assert (layout/'source/00001_boxes.json').resolve()==b.resolve()
    assert not list(layout.rglob('*mask.mp4'))
    assert all(p.is_symlink() for p in layout.rglob('*.mp4'))


def test_fingerprint_changes_when_same_path_video_changes(tmp_path):
    video=tmp_path/'x.mp4';video.write_bytes(b'old')
    mapping=tmp_path/'mapping.json';mapping.write_text(json.dumps([{'generated':str(video)}]))
    a=fingerprint(mapping,tmp_path,[])
    video.write_bytes(b'new-content')
    assert a!=fingerprint(mapping,tmp_path,[])


def test_summary_missing_requested_stage_fails(tmp_path):
    p=tmp_path/'summary.json'
    run=subprocess.run([sys.executable,str(ROOT/'tools/summarize.py'),'--selected','pose','--output',str(p)],capture_output=True)
    assert run.returncode==1
    assert json.loads(p.read_text())['failure_count']['facebench']==1


def test_validation_does_not_allow_one_extra_frame(tmp_path):
    p=tmp_path/'origin.mp4';write_video(p,3)
    g=tmp_path/'gen.mp4';write_video(g,4)
    b=boxes_file(tmp_path,{str(i):[0,0,5,5] for i in range(4)})
    m=tmp_path/'mapping.json';m.write_text(json.dumps([{'case_id':'x','generated':str(g),'ref_video':str(p),'ref_video_facemask':str(b)}]))
    run=subprocess.run([sys.executable,str(ROOT/'tools/validate_mapping.py'),'--mapping',str(m),
        '--output',str(tmp_path/'report.json'),'--errors',str(tmp_path/'errors.log')],capture_output=True)
    assert run.returncode==1
    assert 'ORIGIN_TOO_SHORT' in (tmp_path/'errors.log').read_text()


def test_short_stride_five_stays_inside_first_81_frames(tmp_path, monkeypatch):
    p=tmp_path/'video.mp4';write_video(p,100)
    ref=tmp_path/'ref.png';cv2.imwrite(str(ref),np.full((32,32,3),100,np.uint8))
    b=boxes_file(tmp_path,{str(i):[8,8,24,24] for i in range(100)})
    model=SimpleNamespace(embed=lambda frame:np.array([1.,0.],np.float32))
    row={'name':'case','facebench_video_id':'00001','generated':str(p),'ref_video':str(p),
         'ref_image':str(ref),'ref_video_facemask':str(b)}
    out=strict.evaluate_case(model,row,81,81,42,False,'face-box',5)
    assert out['eval_frame_indices']==list(range(0,81,5))
    assert out['sampled_frame_count']==17
    from swapface_benchmark.metrics.identity_multibackbone import prepare_generated_frames
    _,frames,indices=prepare_generated_frames(row,81,81,42,5)
    assert indices==out['eval_frame_indices'] and len(frames)==17


def test_long_stride_fifteen():
    assert strict.sample_eval_indices(46,0,False,42,15)==[0,15,30,45]


def test_attribute_mapping_needs_neither_reference_nor_gan(tmp_path):
    p=tmp_path/'case.mp4';p.touch()
    b=boxes_file(tmp_path,{'0':[0,0,2,2]})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'cases':[{'case_id':'case','origin_video':str(p),'face_boxes':str(b)}]}))
    mapping=tmp_path/'mapping.json'
    subprocess.run([sys.executable,str(ROOT/'tools/prepare_results.py'),'--manifest',str(manifest),
        '--results-dir',str(tmp_path),'--output',str(mapping),'--no-reference'],check=True,capture_output=True)
    assert json.loads(mapping.read_text())[0]['ref_image'] is None
