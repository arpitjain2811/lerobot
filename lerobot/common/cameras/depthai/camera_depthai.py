# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Provides the DepthAICamera class for capturing frames from Luxonis DepthAI cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any, Dict, List

import cv2
import numpy as np

try:
    import depthai as dai
except Exception as e:
    logging.info(f"Could not import depthai: {e}")

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_depthai import DepthAICameraConfig

logger = logging.getLogger(__name__)


class DepthAICamera(Camera):
    """
    Manages interactions with Luxonis DepthAI cameras for frame and depth recording.

    This class provides an interface similar to `OpenCVCamera` and `RealSenseCamera` but tailored for
    DepthAI devices, leveraging the `depthai` library. It uses the camera's unique MxID for 
    identification, offering more stability than device indices. It also supports capturing 
    depth maps alongside color frames.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    python -m lerobot.find_cameras depthai
    ```

    A `DepthAICamera` instance requires a configuration object specifying the
    camera's MxID or a unique device name. If using the name, ensure only
    one camera with that name is connected.

    The camera's default settings (FPS, resolution, color mode) from the stream
    profile are used unless overridden in the configuration.

    Example:
        ```python
        from lerobot.common.cameras.depthai import DepthAICamera, DepthAICameraConfig
        from lerobot.common.cameras import ColorMode, Cv2Rotation

        # Basic usage with MxID
        config = DepthAICameraConfig(mxid_or_name="14442C10D13EABF200") # Replace with actual MxID
        camera = DepthAICamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example with depth capture and custom settings
        custom_config = DepthAICameraConfig(
            mxid_or_name="14442C10D13EABF200", # Replace with actual MxID
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = DepthAICamera(custom_config)
        depth_camera.connect()

        # Read 1 depth frame
        depth_map = depth_camera.read_depth()

        # Example using a unique camera name
        name_config = DepthAICameraConfig(mxid_or_name="OAK-D-LITE") # If unique
        name_camera = DepthAICamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: DepthAICameraConfig):
        """
        Initializes the DepthAICamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        # Check if mxid_or_name is an MxID (hex string) or a name
        if len(config.mxid_or_name) == 18 and all(c in '0123456789ABCDEF' for c in config.mxid_or_name.upper()):
            self.mxid = config.mxid_or_name
        else:
            self.mxid = self._find_mxid_from_name(config.mxid_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.device: dai.Device | None = None
        self.pipeline: dai.Pipeline | None = None
        self.color_queue: dai.DataOutputQueue | None = None
        self.depth_queue: dai.DataOutputQueue | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.mxid})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera device is connected and pipeline is running."""
        return self.device is not None and self.pipeline is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the DepthAI camera specified in the configuration.

        Initializes the DepthAI device, creates and configures the pipeline with required 
        streams (color and optionally depth), starts the pipeline, and validates the 
        actual stream settings.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g., missing mxid/name, name not unique).
            ConnectionError: If the camera is found but fails to start the pipeline or no DepthAI devices are detected at all.
            RuntimeError: If the pipeline starts but fails to apply requested settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()
            
            # Create color camera node
            cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            cam_rgb.setPreviewSize(self.capture_width or 640, self.capture_height or 480)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            
            if self.fps:
                cam_rgb.setFps(self.fps)

            # Create output queue for color
            color_out = self.pipeline.create(dai.node.XLinkOut)
            color_out.setStreamName("color")
            cam_rgb.preview.link(color_out.input)

            # Create depth camera if requested
            if self.use_depth:
                # Create mono cameras for depth
                mono_left = self.pipeline.create(dai.node.MonoCamera)
                mono_right = self.pipeline.create(dai.node.MonoCamera)
                depth = self.pipeline.create(dai.node.StereoDepth)

                mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
                mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
                mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
                mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

                # Create depth output
                depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
                depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
                depth.setLeftRightCheck(True)
                depth.setSubpixel(False)

                mono_left.out.link(depth.left)
                mono_right.out.link(depth.right)

                depth_out = self.pipeline.create(dai.node.XLinkOut)
                depth_out.setStreamName("depth")
                depth.depth.link(depth_out.input)

            # Connect to device
            device_info = dai.DeviceInfo(self.mxid)
            self.device = dai.Device(self.pipeline, device_info)

            # Get output queues
            self.color_queue = self.device.getOutputQueue(name="color", maxSize=4, blocking=False)
            if self.use_depth:
                self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        except Exception as e:
            self.device = None
            self.pipeline = None
            raise ConnectionError(
                f"Failed to open {self}. "
                "Run `python -m lerobot.find_cameras depthai` to find available cameras."
            ) from e

        self._configure_capture_settings()

        if warmup:
            time.sleep(1)  # DepthAI cameras need time to warm up
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """
        Detects available Luxonis DepthAI cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (MxID), 'name',
            and other available specs, and the default profile properties (width, height, fps).

        Raises:
            OSError: If depthai is not installed.
            ImportError: If depthai is not installed.
        """
        found_cameras_info = []
        
        try:
            devices = dai.Device.getAllAvailableDevices()
            
            for device_info in devices:
                camera_info = {
                    "name": device_info.desc.name,
                    "type": "DepthAI",
                    "id": device_info.getMxId(),
                    "state": device_info.state.name,
                    "protocol": device_info.desc.protocol.name,
                    "platform": device_info.desc.platform.name,
                    "product_name": device_info.desc.productName,
                    "board_name": device_info.desc.boardName,
                    "default_stream_profile": {
                        "format": "RGB888",
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                    },
                }
                found_cameras_info.append(camera_info)
                
        except Exception as e:
            logger.warning(f"Error finding DepthAI cameras: {e}")
            
        return found_cameras_info

    def _find_mxid_from_name(self, name: str) -> str:
        """Finds the MxID for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No DepthAI camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            mxids = [dev["id"] for dev in found_devices]
            raise ValueError(
                f"Multiple DepthAI cameras found with name '{name}'. "
                f"Please use a unique MxID instead. Found MxIDs: {mxids}"
            )

        mxid = str(found_devices[0]["id"])
        return mxid

    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from device stream if not already configured.

        Uses default values if not specified in config. Handles rotation by
        swapping width/height when needed. Original capture dimensions are always stored.

        Raises:
            DeviceNotConnectedError: If device is not connected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")

        if self.fps is None:
            self.fps = 30  # Default FPS for DepthAI

        if self.width is None or self.height is None:
            actual_width = 640  # Default width
            actual_height = 480  # Default height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame (depth) synchronously from the camera.

        This is a blocking call. It waits for a coherent depth frame
        from the camera hardware via the DepthAI pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type `np.uint16` (raw depth values in millimeters) and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        start_time = time.perf_counter()

        try:
            in_depth = self.depth_queue.get()
            depth_frame = in_depth.getFrame()
            
            depth_map_processed = self._postprocess_image(depth_frame, depth_frame=True)

            read_duration_ms = (time.perf_counter() - start_time) * 1e3
            logger.debug(f"{self} read_depth took: {read_duration_ms:.1f}ms")

            return depth_map_processed
            
        except Exception as e:
            raise RuntimeError(f"{self} read_depth failed: {e}")

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a coherent color frame
        from the camera hardware via the DepthAI pipeline.

        Args:
            timeout_ms (int): Maximum time in milliseconds to wait for a frame. Defaults to 200ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        try:
            in_rgb = self.color_queue.get()
            color_frame = in_rgb.getCvFrame()
            
            color_image_processed = self._postprocess_image(color_frame, color_mode)

            read_duration_ms = (time.perf_counter() - start_time) * 1e3
            logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

            return color_image_processed
            
        except Exception as e:
            raise RuntimeError(f"{self} read failed: {e}")

    def _postprocess_image(
        self, image: np.ndarray, color_mode: ColorMode | None = None, depth_frame: bool = False
    ) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from DepthAI).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if color_mode and color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        
        # DepthAI provides RGB by default, convert to BGR if needed
        requested_color_mode = self.color_mode if color_mode is None else color_mode
        if not depth_frame and requested_color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """
        Disconnects from the camera, closes the device, and cleans up resources.

        Stops the background read thread (if running) and closes the DepthAI device.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (device not connected).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.device is not None:
            self.device.close()
            self.device = None
            self.pipeline = None
            self.color_queue = None
            self.depth_queue = None

        logger.info(f"{self} disconnected.")
