# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import _C_ops
from paddle.base.framework import core

from ..framework import in_dynamic_or_pir_mode

__all__ = []

_int_dtype = [
    core.VarDesc.VarType.UINT8,
    core.VarDesc.VarType.INT8,
    core.VarDesc.VarType.INT16,
    core.VarDesc.VarType.INT32,
    core.VarDesc.VarType.INT64,
    core.VarDesc.VarType.BOOL,
]

'''
@dygraph_only
def block_sparse_matmul(x, y, name=None):
    """
    TODO: Add operator's comments details.
    """
    return _C_ops.block_sparse_matmul(x, y)
'''


def sdd_matmul(
    a, b, trans_a, trans_b, trans_c, spdims, block, lut, widths, out=None
):
    print("there is sdd_matmul")
    if in_dynamic_or_pir_mode():
        print("spdims: ", spdims, " type: ", type(spdims))
        print("lut: ", lut, " type: ", type(lut))
        return _C_ops.sdd_matmul(
            a, b, spdims, lut, trans_a, trans_b, trans_c, block, widths, out
        )
    print("not dynamic!!")


def dsd_matmul(
    a, b, trans_a, trans_b, trans_c, spdims, block, lut, widths, out=None
):
    print("there is dsd_matmul")
    # return _C_ops.dsd_matmul(a, b, trans_a, trans_b, trans_c, spdims, block, lut, widths, out)


def dds_matmul(
    a, b, trans_a, trans_b, trans_c, spdims, block, lut, width, out=None
):
    print("there is dds_matmul")
    # return dsd_matmul(b, a, not trans_b, not trans_a, not trans_c, spdims, block, lut, width, out=out)


def sdd_lut(layout, block, device):
    # lut = layout.nonzero(as_tuple=False).to(device).int()
    lut = layout.nonzero(as_tuple=False).to(device).cast('int32')
    # lut = lut.contiguous()
    return lut, None


def dsd_lut(layout, block, step, trans, device):
    '''
    Generates the look-up table for incrementing points in the DSD/DDS matmul.
    Example (BLOCK=32, STEP=16)
    [[1, 0, 0, 1, 0],
     [0, 1, 1, 0, 1],
     [1, 0, 1, 0, 0]]

    Then the offsets for A are
     [0, 16, 32, 48] <- row 0
     \\----/  \\----/
      col=0   col=3
     [64, 80, 96, 112, 128, 144] <- row 1
     \\----/   \\----/  \\----/
      col=1     col=2    col=3
     [160, 176, 192, 208]
    which leads to increments table
    [0, 16, 16, 16, || 64, 16, 16, 16, 16, 16, || 160, 16, 16, 16]

    Because B is dense, the offsets are
    [0, 16, 96, 112] <- row 0
    [32, 48, 64, 80] <- row 1
    [0, 16, 64, 80]  <- row 2
    '''
    sizes = paddle.sum(layout, 2 if trans else 1)
    head_id, col_id = paddle.ones_like(sizes).nonzero(as_tuple=True)
    sizes = sizes.flatten()
    segments = sizes * step

    if trans:
        nnz = layout.nonzero(as_tuple=False)
    else:
        x = paddle.arange(layout.ndim)
        x[1:3] = paddle.to_tensor([2, 1])
        nnz = layout.transpose(perm=x.tolist()).nonzero(as_tuple=False)

    num_blocks = nnz.shape[0]
    offsets = paddle.zeros_like(sizes)
    offsets[1:] = paddle.cumsum(sizes[:-1], axis=0)
    # offsets = paddle.min(offsets, (num_blocks - 1) * paddle.ones_like(offsets))
    offsets = paddle.minimum(
        offsets, (num_blocks - 1) * paddle.ones_like(offsets)
    )

    B_idx = nnz[:, 2] * block
    B_incs = B_idx.clone()
    B_incs[1:] -= B_idx[:-1]
    div = block // step
    # B_incs = B_incs.reshape([-1, 1]).repeat(1, div)
    # B_incs = B_incs.reshape([-1, 1]).expand(shape=[1, div])
    B_incs = B_incs.reshape(shape=[-1, 1])
    B_incs = paddle.concat([B_incs for i in range(div)], axis=1)
    B_incs[:, 1:] = step
    B_incs[:, 0] -= (div - 1) * step

    B_incs[offsets[segments > 0], 0] = B_idx[offsets[segments > 0]]
    # B_incs = B_incs.view(-1)
    B_incs = B_incs.reshape(shape=[-1])

    if trans:
        A_idx = paddle.arange(num_blocks)
    else:
        A_idx = paddle.to_tensor([], dtype='int64')
        current_offset = 0
        # for z in range(layout.size(0)):
        for z in range(layout.shape[0]):
            layoutw = layout[z, :, :].clone().cast('int64')
            msum = layoutw.sum()
            layoutw[layoutw > 0] = 1 + paddle.arange(msum)
            A_idx = paddle.concat(
                (A_idx, current_offset + layoutw.T[layoutw.T > 0] - 1)
            )
            current_offset += msum

    A_incs = A_idx * block * block
    A_incs[1:] -= A_idx[:-1] * block * block
    # A_incs = A_incs.view(-1, 1).repeat(1, div)
    A_incs = A_incs.reshape(shape=[-1, 1])
    A_incs = paddle.concat([A_incs for i in range(div)], axis=1)
    if trans:
        A_incs[:, 1:] = step
        A_incs[:, 0] -= (div - 1) * step
    else:
        A_incs[:, 1:] = step * block
        A_incs[:, 0] -= (div - 1) * step * block
    A_incs[offsets[segments > 0], 0] = A_idx[offsets[segments > 0]]
    # A_incs = A_incs.view(-1)
    A_incs = A_incs.reshape(shape=[-1])

    width = col_id.shape[0]
    offsets = offsets * 2 * div + 4 * width
    segments = segments * div
    offsets = offsets.unsqueeze(axis=1)
    segments = segments.unsqueeze(axis=1)
    header = paddle.stack((offsets, segments, col_id, head_id), axis=1).reshape(
        shape=[-1]
    )  # .contiguous()

    # incs = paddle.stack((B_incs, A_incs), dim=1).view(-1) #.contiguous()
    incs = paddle.stack((B_incs, A_incs), axis=1).reshape(shape=[-1])

    pad = paddle.zeros(20, dtype=incs.dtype)
    incs = paddle.concat((incs, pad))

    lut = paddle.concat((header, incs))
    # lut = lut.type(paddle.int32).to(device)
    lut = lut.cast('int32')

    return lut, width


class matmul:
    def __init__(
        self,
        layout,
        block,
        mode,
        device,
        trans_a=False,
        trans_b=False,
        trans_c=False,
    ):
        if mode not in ['sdd', 'dsd', 'dds']:
            raise NotImplementedError('Supported modes are: sdd, dsd, dds')

        self.fn = {'sdd': sdd_matmul, 'dsd': dsd_matmul, 'dds': dds_matmul}
        self.block = block
        self.mode = mode
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.trans_c = trans_c
        self.layout = layout
        self.spdims = layout.shape
        step = min(block, 32)

        if self.mode == 'sdd':
            self.c_lut, self.c_width = sdd_lut(layout, block, device)
            self.da_lut, self.da_width = dsd_lut(
                layout, block, step, True, device
            )
            self.db_lut, self.db_width = dsd_lut(
                layout, block, step, False, device
            )
        if self.mode == 'dsd':
            self.c_lut, self.c_width = dsd_lut(
                layout, block, step, not self.trans_a, device
            )
            self.da_lut, self.da_width = sdd_lut(layout, block, device)
            self.db_lut, self.db_width = dsd_lut(
                layout, block, step, self.trans_a, device
            )
        if self.mode == 'dds':
            self.c_lut, self.c_width = dsd_lut(
                layout, block, step, self.trans_b, device
            )
            self.da_lut, self.da_width = dsd_lut(
                layout, block, step, not self.trans_b, device
            )
            self.db_lut, self.db_width = sdd_lut(layout, block, device)

    def __call__(self, a, b, out=None):
        # c = _matmul.apply(a, b, self.trans_a, self.trans_b, self.trans_c, self.mode, self.spdims, self.block,
        #        self.c_lut, self.c_width,
        #        self.da_lut, self.da_width,
        #        self.db_lut, self.db_width,
        #        out)
        # return c
        print("call of matmul")
        # c = self.fn[self.mode](a, b, self.trans_a, self.trans_b, self.trans_c, self.mode, self.spdims, self.block,
        #        self.c_lut, self.c_width,
        #        self.da_lut, self.da_width,
        #        self.db_lut, self.db_width,
        #        out)
        c = self.fn[self.mode](
            a,
            b,
            self.trans_a,
            self.trans_b,
            self.trans_c,
            self.spdims,
            self.block,
            self.c_lut,
            self.c_width,
            out,
        )
        return c
