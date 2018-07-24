import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Count how many frames are zeros")
parser.add_argument("--npz", help=".npz file")
args = parser.parse_args()

npzfile = np.load(args.npz)
X = npzfile["X"]
y = npzfile["y"]

cnt = 0

for video_idx in range(X.shape[0]):
    print("Progress: {:2.3%}".format(video_idx / X.shape[0]), end="\r")
    for frame_idx in range(X.shape[1]):
        if not np.any(X[video_idx,frame_idx]):
            cnt = cnt + 1
print("Progress: COMPLETE")

print("Number of zeros: {}/{} ({:2.3%})".format(cnt, X.shape[0] * X.shape[1], cnt/(X.shape[0] * X.shape[1])))
