"""``librknnrt`` over ``ctypes`` -- to split the leak, and then to fix it.

The RSS growth (README, "The bare loop settles it") is 43.8 kB per
``model.infer()`` and nothing else: 273 149 iterations of the identical loop
*without* the inference call move RSS by zero. What that measurement cannot say
is **which** native layer drops the ``free``. The call goes

    kit.runtime.engine.RknnModel.infer
      -> rknnlite.api.RKNNLite.inference          (Python, rknn_lite.py)
           -> RKNNRuntime.set_inputs / .run / .get_outputs
                                                  (Cython, rknn_runtime*.so)
                -> librknnrt.so                   (vendor blob)

and the two candidates -- the Cython extension and ``librknnrt`` -- cannot be
separated from outside: ``gdb`` needs ``ptrace`` on a root process and the image
has no compiler for an ``LD_PRELOAD`` interposer.

This module removes the middle layer instead of instrumenting it. It calls
``librknnrt`` directly, with the same graph and the same input, in four
deliberately different ways:

``get``
    ``rknn_inputs_set`` + ``rknn_run`` + ``rknn_outputs_get`` +
    ``rknn_outputs_release``, ``want_float=1``. Byte-for-byte the API sequence
    the Cython extension performs (its ``.dynstr`` names exactly these five
    entry points and carries the string ``Release outputs failed, ret code:``).
    Flat here means the missing ``free`` is in the extension, and this function
    is already the fix.

``run``
    ``rknn_inputs_set`` + ``rknn_run``, and **nothing else** -- the outputs are
    never fetched. The one call that isolates ``rknn_run`` from the output path:
    if ``get`` leaks and this does not, the defect is in
    ``rknn_outputs_get``/``_release``; if this leaks too, it is in ``rknn_run``
    and no calling convention can avoid it.

``iomem``
    the vendor's own path (``rc_infer.cpp``): ``rknn_create_mem`` +
    ``rknn_set_io_mem`` once at init, then only ``rknn_mem_sync`` +
    ``rknn_run`` + ``rknn_mem_sync`` per frame. There is no per-inference
    allocation at all, so a leak here would have to be inside ``rknn_run``.

``leak``
    the **positive control**, and the reason the other three are believable:
    ``rknn_outputs_get`` with the matching ``rknn_outputs_release`` deliberately
    omitted. This must show a large, obvious climb. A harness that reports
    "flat" for all four has proved nothing except that it cannot see a leak;
    this variant is what demonstrates it can.

Every ``restype``/``argtypes`` below is load-bearing. ctypes types an
unprototyped return as C ``int``, which on aarch64 truncates every returned
pointer to 32 bits -- ``rknn_create_mem`` would hand back a corrupt
``rknn_tensor_mem *`` and the first ``virt_addr`` dereference would take the
interpreter down. The same omission already segfaulted this process once via
``open_memstream`` (see ``esk/mem_probe._malloc_info``).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

LIB_PATH = "/usr/lib/librknnrt.so"

RKNN_SUCC = 0
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

# rknn_query_cmd
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_SDK_VERSION = 3

# rknn_tensor_type / _format
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_NHWC = 1

# rknn_mem_sync_mode
RKNN_MEMORY_SYNC_TO_DEVICE = 0x1
RKNN_MEMORY_SYNC_FROM_DEVICE = 0x2

# aarch64 is LP64, so the header's non-``__arm__`` branch applies.
rknn_context = ctypes.c_uint64


class RknnTensorAttr(ctypes.Structure):
    """``rknn_tensor_attr``. Field order and types are the header's, verbatim.

    ``fl`` (int8) sitting in front of ``zp`` (int32) is the one place a hand-
    packed layout would go wrong; ctypes inserts the same three padding bytes
    the C compiler does, so the struct must not be declared ``_pack_``-ed.
    """

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RknnTensorMem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p),
        ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32),
        ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]


class RknnSdkVersion(ctypes.Structure):
    _fields_ = [("api_version", ctypes.c_char * 256),
                ("drv_version", ctypes.c_char * 256)]


_TYPE_NAMES = {
    0: "float32", 1: "float16", 2: "int8", 3: "uint8", 4: "int16",
    5: "uint16", 6: "int32", 7: "uint32", 8: "int64", 9: "bool",
    10: "int4", 11: "bfloat16",
}
_FMT_NAMES = {0: "NCHW", 1: "NHWC", 2: "NC1HWC2", 3: "UNDEFINED"}

_lib = None


def _load():
    """dlopen ``librknnrt`` once and prototype every entry point we use."""
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)

    lib.rknn_init.restype = ctypes.c_int
    lib.rknn_init.argtypes = [
        ctypes.POINTER(rknn_context), ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p,
    ]
    lib.rknn_destroy.restype = ctypes.c_int
    lib.rknn_destroy.argtypes = [rknn_context]

    lib.rknn_query.restype = ctypes.c_int
    lib.rknn_query.argtypes = [rknn_context, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_uint32]

    lib.rknn_inputs_set.restype = ctypes.c_int
    lib.rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32,
                                    ctypes.POINTER(RknnInput)]

    lib.rknn_run.restype = ctypes.c_int
    lib.rknn_run.argtypes = [rknn_context, ctypes.c_void_p]

    lib.rknn_outputs_get.restype = ctypes.c_int
    lib.rknn_outputs_get.argtypes = [rknn_context, ctypes.c_uint32,
                                     ctypes.POINTER(RknnOutput),
                                     ctypes.c_void_p]
    lib.rknn_outputs_release.restype = ctypes.c_int
    lib.rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32,
                                         ctypes.POINTER(RknnOutput)]

    # Pointer returns: without these restypes aarch64 truncates them to 32 bits.
    lib.rknn_create_mem.restype = ctypes.POINTER(RknnTensorMem)
    lib.rknn_create_mem.argtypes = [rknn_context, ctypes.c_uint32]
    lib.rknn_destroy_mem.restype = ctypes.c_int
    lib.rknn_destroy_mem.argtypes = [rknn_context, ctypes.POINTER(RknnTensorMem)]
    lib.rknn_set_io_mem.restype = ctypes.c_int
    lib.rknn_set_io_mem.argtypes = [rknn_context, ctypes.POINTER(RknnTensorMem),
                                    ctypes.POINTER(RknnTensorAttr)]
    lib.rknn_mem_sync.restype = ctypes.c_int
    lib.rknn_mem_sync.argtypes = [rknn_context, ctypes.POINTER(RknnTensorMem),
                                  ctypes.c_int]
    _lib = lib
    return lib


def _attr_dict(attr: RknnTensorAttr) -> dict:
    return {
        "index": attr.index,
        "name": attr.name.decode("utf-8", "replace"),
        "dims": [attr.dims[i] for i in range(attr.n_dims)],
        "n_elems": attr.n_elems,
        "size": attr.size,
        "size_with_stride": attr.size_with_stride,
        "fmt": _FMT_NAMES.get(attr.fmt, attr.fmt),
        "type": _TYPE_NAMES.get(attr.type, attr.type),
        "zp": attr.zp,
        "scale": attr.scale,
    }


class CtypesRknnModel:
    """One ``rknn_context``, driven straight from Python.

    Mirrors ``kit.runtime.engine.RknnModel``'s surface (``infer``/``release``/
    context manager) so the same bench loop can drive either, and so the class
    can be dropped into the app once a variant is shown to be flat.

    ``mode`` selects which of the four API sequences ``infer`` performs; see the
    module docstring. Only ``iomem`` allocates anything per model -- the other
    three reuse one ``rknn_input`` and one ``rknn_output`` array for the life of
    the object, so the harness itself allocates nothing per call.
    """

    def __init__(self, path: str, mode: str = "get"):
        if mode not in ("get", "run", "iomem", "leak"):
            raise ValueError(f"unknown ctypes infer mode {mode!r}")
        self.path = path
        self.mode = mode
        self.lib = _load()
        self.ctx = rknn_context(0)
        self._mems: list = []
        self._released = False

        with open(path, "rb") as fh:
            blob = fh.read()
        # Kept alive for the object's life. The vendor frees its copy right
        # after rknn_init (rc_infer.cpp:845 "free model_data after rknn_init"),
        # so holding it is belt-and-braces rather than required -- but a freed
        # buffer that the runtime did keep a pointer into is not a failure mode
        # worth discovering during a 20-minute leak run.
        self._blob = ctypes.create_string_buffer(blob, len(blob))
        ret = self.lib.rknn_init(ctypes.byref(self.ctx), self._blob,
                                 len(blob), 0, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_init failed for {path!r}: ret={ret}")

        ver = RknnSdkVersion()
        if self.lib.rknn_query(self.ctx, RKNN_QUERY_SDK_VERSION,
                               ctypes.byref(ver), ctypes.sizeof(ver)) == RKNN_SUCC:
            self.sdk = {"api": ver.api_version.decode("utf-8", "replace"),
                        "drv": ver.drv_version.decode("utf-8", "replace")}
        else:
            self.sdk = {}

        io = RknnInputOutputNum()
        ret = self.lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM,
                                  ctypes.byref(io), ctypes.sizeof(io))
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_query(IN_OUT_NUM) failed: ret={ret}")
        self.n_input = int(io.n_input)
        self.n_output = int(io.n_output)

        self.input_attrs = (RknnTensorAttr * self.n_input)()
        for i in range(self.n_input):
            self.input_attrs[i].index = i
            ret = self.lib.rknn_query(
                self.ctx, RKNN_QUERY_INPUT_ATTR,
                ctypes.byref(self.input_attrs[i]),
                ctypes.sizeof(RknnTensorAttr),
            )
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(INPUT_ATTR {i}) failed: ret={ret}")

        self.output_attrs = (RknnTensorAttr * self.n_output)()
        for i in range(self.n_output):
            self.output_attrs[i].index = i
            ret = self.lib.rknn_query(
                self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                ctypes.byref(self.output_attrs[i]),
                ctypes.sizeof(RknnTensorAttr),
            )
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(OUTPUT_ATTR {i}) failed: ret={ret}")

        # Reused across calls; see class docstring.
        self._inputs = (RknnInput * self.n_input)()
        self._outputs = (RknnOutput * self.n_output)()
        # Output shapes, cached once: rknnlite hands back NCHW float32 with the
        # attr's dims, and the app's head decode is written against that.
        self._out_shapes = [
            tuple(self.output_attrs[i].dims[d]
                  for d in range(self.output_attrs[i].n_dims))
            for i in range(self.n_output)
        ]

        if mode == "iomem":
            self._setup_iomem()

    # ------------------------------------------------------------ zero copy

    def _setup_iomem(self):
        """``rknn_create_mem`` + ``rknn_set_io_mem`` once, exactly as rc_infer.

        The attribute *mutation* below is not decoration -- it is what tells the
        runtime how to interpret the buffer we hand it. ``rc_infer.cpp`` rewrites
        ``input_attrs[0].type``/``.fmt`` to UINT8/NHWC and every
        ``output_attrs[i].type`` to INT8 **before** the matching
        ``rknn_set_io_mem``. Passing the queried attrs unmodified would bind the
        memory under the graph's native description (often float32 NC1HWC2) and
        the runtime would size and convert against that instead.

        The two sizes differ on purpose, and both are the vendor's:
        the input uses ``size_with_stride`` (a width padded to the NPU stride
        needs the padded buffer; a short one is a silent out-of-bounds DMA, not
        an error return), the outputs use ``n_elems * sizeof(int8_t)``, because
        after the type override the output is dense int8.
        """
        lib = self.lib
        attr = self.input_attrs[0]
        attr.type = RKNN_TENSOR_UINT8
        attr.fmt = RKNN_TENSOR_NHWC
        in_size = attr.size_with_stride or attr.size
        if not in_size:
            raise RuntimeError("invalid input mem size (both size fields zero)")
        mem = lib.rknn_create_mem(self.ctx, in_size)
        if not mem:
            raise RuntimeError("rknn_create_mem(input) returned NULL")
        ret = lib.rknn_set_io_mem(self.ctx, mem, ctypes.byref(attr))
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_set_io_mem(input) failed: ret={ret}")
        self._in_mem = mem
        self._in_size = in_size
        self._mems.append(mem)

        self._out_mems = []
        for i in range(self.n_output):
            oattr = self.output_attrs[i]
            oattr.type = RKNN_TENSOR_INT8
            osize = oattr.n_elems  # sizeof(int8_t) == 1
            omem = lib.rknn_create_mem(self.ctx, osize)
            if not omem:
                raise RuntimeError(f"rknn_create_mem(output {i}) returned NULL")
            ret = lib.rknn_set_io_mem(self.ctx, omem, ctypes.byref(oattr))
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_set_io_mem(output {i}) failed: ret={ret}")
            self._out_mems.append(omem)
            self._mems.append(omem)

    # ---------------------------------------------------------------- infer

    def _prepare_input(self, input_uint8):
        arr = np.asarray(input_uint8)
        if arr.ndim == 3:
            arr = np.expand_dims(arr, 0)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr)

    def infer(self, input_uint8):
        """One forward pass, by whichever API sequence ``mode`` names."""
        if self.mode == "iomem":
            return self._infer_iomem(input_uint8)
        return self._infer_outputs(input_uint8)

    def _infer_outputs(self, input_uint8):
        lib = self.lib
        arr = self._prepare_input(input_uint8)
        attr = self.input_attrs[0]

        inp = self._inputs[0]
        inp.index = 0
        inp.buf = arr.ctypes.data_as(ctypes.c_void_p)
        inp.size = arr.nbytes
        inp.pass_through = 0
        inp.type = RKNN_TENSOR_UINT8
        inp.fmt = RKNN_TENSOR_NHWC
        ret = lib.rknn_inputs_set(self.ctx, self.n_input, self._inputs)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_inputs_set failed: ret={ret}")

        ret = lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed: ret={ret}")

        if self.mode == "run":
            # Deliberately no outputs_get. Isolates rknn_run; see module doc.
            return None

        for i in range(self.n_output):
            self._outputs[i].want_float = 1
            self._outputs[i].is_prealloc = 0
            self._outputs[i].index = i
            self._outputs[i].buf = None
            self._outputs[i].size = 0
        ret = lib.rknn_outputs_get(self.ctx, self.n_output, self._outputs, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_outputs_get failed: ret={ret}")

        out = []
        for i in range(self.n_output):
            o = self._outputs[i]
            # Copy before release: the header is explicit that after
            # rknn_outputs_release the buf pointer is freed and must not be
            # used. np.frombuffer on the raw pointer would alias it.
            buf = ctypes.string_at(o.buf, o.size)
            a = np.frombuffer(buf, dtype=np.float32)
            shape = self._out_shapes[i]
            if a.size == int(np.prod(shape)):
                a = a.reshape(shape)
            out.append(a)

        if self.mode != "leak":
            ret = lib.rknn_outputs_release(self.ctx, self.n_output, self._outputs)
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_outputs_release failed: ret={ret}")
        return out

    def _infer_iomem(self, input_uint8):
        lib = self.lib
        arr = self._prepare_input(input_uint8)
        mem = self._in_mem.contents
        n = min(arr.nbytes, mem.size)
        ctypes.memmove(mem.virt_addr, arr.ctypes.data_as(ctypes.c_void_p), n)
        ret = lib.rknn_mem_sync(self.ctx, self._in_mem, RKNN_MEMORY_SYNC_TO_DEVICE)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_mem_sync(input) failed: ret={ret}")

        ret = lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed: ret={ret}")

        out = []
        for i, omem in enumerate(self._out_mems):
            ret = lib.rknn_mem_sync(self.ctx, omem, RKNN_MEMORY_SYNC_FROM_DEVICE)
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_mem_sync(output {i}) failed: ret={ret}")
            m = omem.contents
            # Raw quantized bytes. The zero-copy path returns the tensor in the
            # model's native type and layout; dequant is the caller's job and is
            # deliberately NOT done here, because the question this variant
            # answers is about allocation, not about numerics.
            out.append(np.frombuffer(ctypes.string_at(m.virt_addr, m.size),
                                     dtype=np.int8))
        return out

    # -------------------------------------------------------------- teardown

    def describe(self) -> dict:
        return {
            "lib": LIB_PATH,
            "path": self.path,
            "mode": self.mode,
            "sdk": self.sdk,
            "n_input": self.n_input,
            "n_output": self.n_output,
            "inputs": [_attr_dict(self.input_attrs[i]) for i in range(self.n_input)],
            "outputs": [_attr_dict(self.output_attrs[i]) for i in range(self.n_output)],
            "pid": os.getpid(),
        }

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            for mem in self._mems:
                self.lib.rknn_destroy_mem(self.ctx, mem)
            self._mems = []
            if self.ctx.value:
                self.lib.rknn_destroy(self.ctx)
                self.ctx = rknn_context(0)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
