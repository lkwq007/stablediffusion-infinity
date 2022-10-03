"""
https://github.com/Trinkle23897/Fast-Poisson-Image-Editing
MIT License

Copyright (c) 2022 Jiayi Weng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import argparse
import os
import fpie
from process import ALL_BACKEND, CPU_COUNT, DEFAULT_BACKEND
from fpie.io import read_images, write_image
from process import BaseProcessor, EquProcessor, GridProcessor

from PIL import Image
import numpy as np
import skimage
import skimage.measure

class PhotometricCorrection:
    def __init__(self):
        self.get_parser("cli")
        args=self.parser.parse_args(["--method","grid","-g","src","-s","a","-t","a","-o","a"])
        self.args=args
        proc: BaseProcessor
        proc = GridProcessor(
            args.gradient,
            args.backend,
            args.cpu,
            args.mpi_sync_interval,
            args.block_size,
            args.grid_x,
            args.grid_y,
        )
        print(
            f"Successfully initialize PIE {args.method} solver "
            f"with {args.backend} backend"
        )
        self.proc=proc

    def run(self, original_image, inpainted_image):
        input_arr=np.array(original_image)
        output_arr=np.array(inpainted_image)
        mask=input_arr[:,:,3:4]
        mask=255-mask
        mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
        mask = mask.repeat(8, axis=0).repeat(8, axis=1)
        src = inpainted_image[:,:,0:3]
        tgt = src.copy()
        src, mask, tgt=0,0,0
        proc=self.proc
        args=self.args
        if proc.root:
            n = proc.reset(src, mask, tgt, (args.h0, args.w0), (args.h1, args.w1))
        proc.sync()
        if proc.root:
            result = tgt
        if args.p == 0:
            args.p = args.n

        for i in range(0, args.n, args.p):
            if proc.root:
                result, err = proc.step(args.p)  # type: ignore
            else:
                proc.step(args.p)

        if proc.root:
            return result


    def get_parser(self,gen_type: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-v", "--version", action="store_true", help="show the version and exit"
        )
        parser.add_argument(
            "--check-backend", action="store_true", help="print all available backends"
        )
        if gen_type == "gui" and "mpi" in ALL_BACKEND:
            # gui doesn't support MPI backend
            ALL_BACKEND.remove("mpi")
        parser.add_argument(
            "-b",
            "--backend",
            type=str,
            choices=ALL_BACKEND,
            default=DEFAULT_BACKEND,
            help="backend choice",
        )
        parser.add_argument(
            "-c",
            "--cpu",
            type=int,
            default=CPU_COUNT,
            help="number of CPU used",
        )
        parser.add_argument(
            "-z",
            "--block-size",
            type=int,
            default=1024,
            help="cuda block size (only for equ solver)",
        )
        parser.add_argument(
            "--method",
            type=str,
            choices=["equ", "grid"],
            default="equ",
            help="how to parallelize computation",
        )
        parser.add_argument("-s", "--source", type=str, help="source image filename")
        if gen_type == "cli":
            parser.add_argument(
                "-m",
                "--mask",
                type=str,
                help="mask image filename (default is to use the whole source image)",
                default="",
            )
        parser.add_argument("-t", "--target", type=str, help="target image filename")
        parser.add_argument("-o", "--output", type=str, help="output image filename")
        if gen_type == "cli":
            parser.add_argument(
                "-h0", type=int, help="mask position (height) on source image", default=0
            )
            parser.add_argument(
                "-w0", type=int, help="mask position (width) on source image", default=0
            )
            parser.add_argument(
                "-h1", type=int, help="mask position (height) on target image", default=0
            )
            parser.add_argument(
                "-w1", type=int, help="mask position (width) on target image", default=0
            )
        parser.add_argument(
            "-g",
            "--gradient",
            type=str,
            choices=["max", "src", "avg"],
            default="max",
            help="how to calculate gradient for PIE",
        )
        parser.add_argument(
            "-n",
            type=int,
            help="how many iteration would you perfer, the more the better",
            default=5000,
        )
        if gen_type == "cli":
            parser.add_argument(
                "-p", type=int, help="output result every P iteration", default=0
            )
        if "mpi" in ALL_BACKEND:
            parser.add_argument(
                "--mpi-sync-interval",
                type=int,
                help="MPI sync iteration interval",
                default=100,
            )
        parser.add_argument(
            "--grid-x", type=int, help="x axis stride for grid solver", default=8
        )
        parser.add_argument(
            "--grid-y", type=int, help="y axis stride for grid solver", default=8
        )
        self.parser=parser

