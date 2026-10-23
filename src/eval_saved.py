"""train.py --save-models 로 저장한 모델을 다른 추출 결과(예: 화면비를 바꿔 다시 뽑은 좌표)에 적용해 정확도를 잰다.

사용: python src/eval_saved.py <모델.pt> <특징 폴더>... [--signers 1,2,3]
--signers 를 안 주면 모델이 저장될 때의 시험 수행자를 쓴다. 전처리 설정은 .pt 안의 config 를 그대로 따른다.
"""
import argparse

import numpy as np
import torch

import dataset as ds
import models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("features", nargs="+")
    ap.add_argument("--signers", default="")
    a = ap.parse_args()
    ck = torch.load(a.model, weights_only=False)
    cfg, classes = ck["config"], ck["classes"]
    signers = [int(s) for s in a.signers.split(",")] if a.signers else ck["split"]["test"]
    clips = [c for c in ds.load_clips(a.features, classes) if c["signer"] in signers]
    X, y, sg = ds.to_arrays(clips, classes, cfg["length"], cfg["hand_rel"], angles=cfg["angles"], traj=cfg["traj"],
                            bones=cfg["bones"], origin=cfg["origin"], hand_z=cfg["hand_z"], folds=cfg["folds"])
    model = models.build(ck["model"], ck["n_feat"], len(classes))
    model.load_state_dict(ck["state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy().argmax(1)
    print("클립 %d개, 수행자 %s, 정확도 %.4f" % (len(y), signers, (pred == y).mean()))
    for s in sorted(set(sg.tolist())):
        m = sg == s
        print("  수행자 %d: %.4f (%d개)" % (s, (pred[m] == y[m]).mean(), m.sum()))


if __name__ == "__main__":
    main()
