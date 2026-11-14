import dxcam

from iwmymdosr.dataset.datasets.objects import Meta
from iwmymdosr.iwmymdosr.runners.base_runner import BaseRunner
from iwmymdosr.model.model import Iwmymdosr
import pyautogui as pag

class Runner(BaseRunner):
    def __init__(self, model: Iwmymdosr, meta: Meta, device):
        super().__init__(model, meta, device)

        pag.PAUSE = 0
        pag.FAILSAFE = False

        self.camera = dxcam.create(output_color="BGR")
        self.camera.start(target_fps=60)

    def __del__(self):
        self.camera.stop()

    def get_current_resolution(self) -> tuple[int, int]:
        return (self.camera.width, self.camera.height)

    def get_frame(self):
        return self.camera.get_latest_frame()
    
    def move_cursor(self, d):
        pag.moveTo(d[0], d[1])

    def get_mouse_position(self):
        pos = pag.position()
        return (pos[0], pos[1])