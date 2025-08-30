import av
import cv2 as cv
import cv2
import os
import concurrent.futures
import numpy as np

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    print("PyOpenCL not available, falling back to CPU processing")

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
    self.cap.set(cv.CAP_PROP_FPS, 25.0)

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      # Rotate the frame 180 degrees (flip both axes)
      frame = cv.flip(frame, -1)
      yuv = Camera.bgr2nv12(frame)
      yield yuv.data.tobytes()
    self.cap.release()


class CameraMJPG:
    def __init__(self, cam_type_state, stream_type, camera_id):
        try:
            camera_id = int(camera_id)
        except ValueError:
            pass

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"faield to open video{camera_id}")

        self._configure_camera_format("MJPG")
        actual_format = self._get_current_format()
        print("format: ", actual_format)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"fps: {self.fps}")

        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"width: {self.W} hight: {self.H}")

        self.cur_frame_id = 0
        self.cam_type_state = cam_type_state
        self.stream_type = stream_type
        self.current_format = actual_format

        # single_thread, multi_thread, opencl
        self.processing_mode = os.getenv("MJPG_PROCESSING_MODE", "multi_thread")
        self.num_threads = int(os.getenv("MJPG_THREADS", "4"))

        print(f"mjpeg encoding mode: {self.processing_mode}")
        if self.processing_mode == "multi_thread":
            print(f"encoding thread count: {self.num_threads}")

        self._init_processing_resources()

    def _configure_camera_format(self, target_fourcc):
        fourcc = cv2.VideoWriter_fourcc(*target_fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 20)

    def _get_current_format(self):
        fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])

    def _init_processing_resources(self):
        self.thread_pool = None
        self.opencl_available = False

        if self.processing_mode == "multi_thread":
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads
            )
            print(f"init encoding threadpool count: {self.num_threads}")
        elif self.processing_mode == "opencl":
            self._init_opencl()

    def _init_opencl(self):
        if not OPENCL_AVAILABLE:
            print("opencl unavailable. defaulting to use CPU encoding")
            self.processing_mode = "single_thread"
            return

        try:
            self.ctx = cl.create_some_context()
            self.queue = cl.CommandQueue(self.ctx)
            self._setup_opencl_kernels()
            self.opencl_available = True
        except Exception as e:
            print(f"failed to init opencl. defaulting to use CPU encoding: {e}")
            self.processing_mode = "single_thread"
            self.opencl_available = False

    def _setup_opencl_kernels(self):
        # rgb_to_nv12.cl
        kernel_source = f"""
        #define RGB_TO_Y(r, g, b) ((((mul24(b, 13) + mul24(g, 65) + mul24(r, 33)) + 64) >> 7) + 16)
        #define RGB_TO_U(r, g, b) ((mul24(b, 56) - mul24(g, 37) - mul24(r, 19) + 0x8080) >> 8)
        #define RGB_TO_V(r, g, b) ((mul24(r, 56) - mul24(g, 47) - mul24(b, 9) + 0x8080) >> 8)
        #define AVERAGE(x, y, z, w) ((convert_ushort(x) + convert_ushort(y) + convert_ushort(z) + convert_ushort(w) + 1) >> 1)

        __kernel void bgr_to_nv12(__global uchar const * const bgr,
                                  __global uchar * out_yuv)
        {{
            const int dx = get_global_id(0);
            const int dy = get_global_id(1);
            const int col = dx * 2;
            const int row = dy * 2;

            if (col >= {self.W} || row >= {self.H}) return;

            const int bgr_stride = {self.W} * 3;
            const int y_size = {self.W} * {self.H};

            // 处理 2x2 像素块
            for (int r = 0; r < 2 && (row + r) < {self.H}; r++) {{
                for (int c = 0; c < 2 && (col + c) < {self.W}; c++) {{
                    int bgr_idx = (row + r) * bgr_stride + (col + c) * 3;
                    int y_idx = (row + r) * {self.W} + (col + c);

                    uchar b = bgr[bgr_idx];
                    uchar g = bgr[bgr_idx + 1];
                    uchar r_val = bgr[bgr_idx + 2];

                    out_yuv[y_idx] = RGB_TO_Y(r_val, g, b);
                }}
            }}

            // UV 分量 (2x2 块的平均值)
            if (row % 2 == 0 && col % 2 == 0) {{
                int uv_row = row / 2;
                int uv_col = col / 2;
                int uv_idx = y_size + uv_row * {self.W} + uv_col * 2;

                // 获取 2x2 块的像素值
                int bgr_idx_00 = row * bgr_stride + col * 3;
                int bgr_idx_01 = row * bgr_stride + (col + 1) * 3;
                int bgr_idx_10 = (row + 1) * bgr_stride + col * 3;
                int bgr_idx_11 = (row + 1) * bgr_stride + (col + 1) * 3;

                if ((col + 1) < {self.W} && (row + 1) < {self.H}) {{
                    uchar b_avg = (bgr[bgr_idx_00] + bgr[bgr_idx_01] + bgr[bgr_idx_10] + bgr[bgr_idx_11]) / 4;
                    uchar g_avg = (bgr[bgr_idx_00 + 1] + bgr[bgr_idx_01 + 1] + bgr[bgr_idx_10 + 1] + bgr[bgr_idx_11 + 1]) / 4;
                    uchar r_avg = (bgr[bgr_idx_00 + 2] + bgr[bgr_idx_01 + 2] + bgr[bgr_idx_10 + 2] + bgr[bgr_idx_11 + 2]) / 4;

                    out_yuv[uv_idx] = RGB_TO_U(r_avg, g_avg, b_avg);
                    out_yuv[uv_idx + 1] = RGB_TO_V(r_avg, g_avg, b_avg);
                }}
            }}
        }}
        """

        self.program = cl.Program(self.ctx, kernel_source).build()
        self.bgr_to_nv12_kernel = self.program.bgr_to_nv12

    @staticmethod
    def _bgr_to_nv12(bgr_frame):
        frame = av.VideoFrame.from_ndarray(bgr_frame, format='bgr24')
        return frame.reformat(format='nv12').to_ndarray().data.tobytes()

    def _bgr_to_nv12_single(self, frame):
        return self._bgr_to_nv12(frame)

    def _process_frame_chunk(self, frame):
        return self._bgr_to_nv12(frame)

    def _bgr_to_nv12_multi(self, frame):
        if self.thread_pool is None:
            return self._bgr_to_nv12_single(frame)

        future = self.thread_pool.submit(self._process_frame_chunk, frame)
        return future.result()

    def _bgr_to_nv12_opencl(self, frame):
        if not self.opencl_available:
            return self._bgr_to_nv12_single(frame)

        try:
            frame_flat = frame.flatten().astype(np.uint8)
            nv12_size = self.W * self.H * 3 // 2

            frame_cl = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=frame_flat)
            result_cl = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, nv12_size)

            global_size = (self.W // 2, self.H // 2)
            self.bgr_to_nv12_kernel(self.queue, global_size, None, frame_cl, result_cl)

            result = np.empty(nv12_size, dtype=np.uint8)
            cl.enqueue_copy(self.queue, result, result_cl).wait()

            return result.tobytes()
        except Exception as e:
            print(f"opencl encode failed. use CPU encoding: {e}")
            return self._bgr_to_nv12_single(frame)

    def read_frames(self):
        """ read and convert to NV12"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.current_format == "MJPG":
                if frame.shape != (self.H, self.W, 3):
                    raise ValueError("MJPG data invalid. w/h missmatched")

            if self.processing_mode == "single_thread":
                yield self._bgr_to_nv12_single(frame)
            elif self.processing_mode == "multi_thread":
                yield self._bgr_to_nv12_multi(frame)
            elif self.processing_mode == "opencl":
                yield self._bgr_to_nv12_opencl(frame)
            else:
                yield self._bgr_to_nv12_single(frame)

        self.cap.release()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

