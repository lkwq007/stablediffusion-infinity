from PIL import Image
from PIL import ImageFilter
import cv2
import numpy as np
import scipy
import scipy.signal
from scipy.spatial import cKDTree

import os
from perlin2d import *

patch_match_compiled = True

try:
    from PyPatchMatch import patch_match
except Exception as e:
    try:
        import patch_match
    except Exception as e:
        patch_match_compiled = False

try:
    patch_match
except NameError:
    print("patch_match compiling failed, will fall back to edge_pad")
    patch_match_compiled = False




def edge_pad(img, mask, mode=1):
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, mask


def perlin_noise(img, mask):
    lin = np.linspace(0, 5, mask.shape[0], endpoint=False)
    x, y = np.meshgrid(lin, lin)
    avg = img.mean(axis=0).mean(axis=0)
    # noise=[((perlin(x, y)+1)*128+avg[i]).astype(np.uint8) for i in range(3)]
    noise = [((perlin(x, y) + 1) * 0.5 * 255).astype(np.uint8) for i in range(3)]
    noise = np.stack(noise, axis=-1)
    # mask=skimage.measure.block_reduce(mask,(8,8),np.min)
    # mask=mask.repeat(8, axis=0).repeat(8, axis=1)
    # mask_image=Image.fromarray(mask)
    # mask_image=mask_image.filter(ImageFilter.GaussianBlur(radius = 4))
    # mask=np.array(mask_image)
    nmask = mask.copy()
    # nmask=nmask/255.0
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    # img=img.astype(np.uint8)
    return img, mask


def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask


def cv2_telea(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_NS)
    return ret, mask


def patch_match_func(img, mask):
    ret = patch_match.inpaint(img, mask=255 - mask, patch_size=3)
    return ret, mask


def mean_fill(img, mask):
    avg = img.mean(axis=0).mean(axis=0)
    img[mask < 1] = avg
    return img, mask


functbl = {
    "gaussian": gaussian_noise,
    "perlin": perlin_noise,
    "edge_pad": edge_pad,
    "patchmatch": patch_match_func if patch_match_compiled else edge_pad,
    "cv2_ns": cv2_ns,
    "cv2_telea": cv2_telea,
    "mean_fill": mean_fill,
}

try:
    from postprocess import PhotometricCorrection
    correction_func = PhotometricCorrection()
except Exception as e:
    print(e, "so PhotometricCorrection is disabled")
    class DummyCorrection:
        def __init__(self):
            self.backend=""
            pass
        def run(self,a,b,**kwargs):
            return b
    correction_func=DummyCorrection()

if "taichi" in correction_func.backend:
    import sys
    import io
    import base64
    from PIL import Image
    def base64_to_pil(base64_str):
        data = base64.b64decode(str(base64_str))
        pil = Image.open(io.BytesIO(data))
        return pil

    def pil_to_base64(out_pil):
        out_buffer = io.BytesIO()
        out_pil.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        base64_bytes = base64.b64encode(out_buffer.read())
        base64_str = base64_bytes.decode("ascii")
        return base64_str
    from subprocess import Popen, PIPE, STDOUT
    class SubprocessCorrection:
        def __init__(self):
            self.backend=correction_func.backend
            self.child= Popen(["python", "postprocess.py"], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        def run(self,img_input,img_inpainted,mode):
            base64_str_input = pil_to_base64(img_input)
            base64_str_inpainted = pil_to_base64(img_inpainted)
            self.child.stdin.write(f"{base64_str_input},{base64_str_inpainted},{mode}\n".encode())
            self.child.stdin.flush()
            out = self.child.stdout.readline()
            base64_str=out.decode().strip()
            return base64_to_pil(base64_str)
