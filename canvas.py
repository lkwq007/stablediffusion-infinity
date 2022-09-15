from ipycanvas import Canvas, hold_canvas, MultiCanvas
import ipywidgets as widgets
import numpy as np

PAINT_SELECTION = "✥"
IMAGE_SELECTION = "🖼️"
NOP_MODE = 0
PAINT_MODE = 1
IMAGE_MODE = 2

# RGBA for masking
class InfCanvas:
    def __init__(
        self,
        width,
        height,
        selection_size=256,
        grid_size=32,
        patch_size=4096,
        test_mode=False,
    ) -> None:
        assert selection_size < min(height, width)
        self.width = width
        self.height = height
        self.canvas = MultiCanvas(4, width=width, height=height)
        # self.canvas = Canvas(width=width, height=height)
        self.view_pos = [0, 0]
        self.cursor = [
            width // 2 - selection_size // 2,
            height // 2 - selection_size // 2,
        ]
        self.data = {}
        self.grid_size = grid_size
        self.selection_size = selection_size
        self.patch_size = patch_size
        # note that for image data, the height comes before width
        self.buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.sel_buffer = np.zeros((selection_size, selection_size, 4), dtype=np.uint8)
        self.sel_buffer_bak = np.zeros(
            (selection_size, selection_size, 4), dtype=np.uint8
        )
        self.sel_dirty = False
        self.buffer_dirty = False
        self.mouse_pos = [-1, -1]
        self.mouse_state = 0
        self.output = widgets.Output()
        self.test_mode = test_mode
        self.buffer_updated = False
        self.image_move_freq = 5
        # inpaint pipeline from diffuser

    def setup_mouse(self):
        self.image_move_cnt = 0

        def get_mouse_mode():
            if self.mode_button.value == PAINT_SELECTION:
                return PAINT_MODE
            return IMAGE_MODE

        def handle_mouse_down(x, y):
            self.mouse_state = get_mouse_mode()

        def handle_mouse_out():
            last_state = self.mouse_state
            self.mouse_state = NOP_MODE
            self.image_move_cnt = 0
            if last_state == IMAGE_MODE:
                with hold_canvas():
                    self.clear_background()
                    self.draw_buffer()
                    self.canvas[2].clear()
                    self.draw_selection_box()

        def handle_mouse_up(x, y):
            last_state = self.mouse_state
            self.mouse_state = NOP_MODE
            self.image_move_cnt = 0
            if last_state == IMAGE_MODE:
                with hold_canvas():
                    self.clear_background()
                    self.draw_buffer()
                    self.canvas[2].clear()
                    self.draw_selection_box()

        def handle_mouse_move(x, y):
            x0, y0 = self.mouse_pos
            xo = x - x0
            yo = y - y0
            if self.mouse_state == PAINT_MODE:
                with self.output:
                    self.update_cursor(int(xo), int(yo))
                with hold_canvas():
                    # self.clear_background()
                    if self.buffer_updated:
                        self.draw_buffer()
                        self.buffer_updated = False
                    self.draw_selection_box()
            elif self.mouse_state == IMAGE_MODE:
                self.image_move_cnt += 1
                with self.output:
                    self.update_view_pos(int(xo), int(yo))
                if self.image_move_cnt == self.image_move_freq:
                    with hold_canvas():
                        self.clear_background()
                        self.draw_buffer()
                        self.canvas[2].clear()
                        self.draw_selection_box()
                    self.image_move_cnt = 0
            self.mouse_pos[0] = x
            self.mouse_pos[1] = y

        self.canvas.on_mouse_down(handle_mouse_down)
        self.canvas.on_mouse_move(handle_mouse_move)
        self.canvas.on_mouse_up(handle_mouse_up)
        self.canvas.on_mouse_out(handle_mouse_out)

    def setup_widgets(self):
        self.mode_button = widgets.ToggleButtons(
            options=[PAINT_SELECTION, IMAGE_SELECTION],
            disabled=False,
            button_style="",
            style={"button_width": "50px", "font_weight": "bold"},
            tooltips=["Outpaint region", "Image"],
        )
        self.test_button = widgets.ToggleButtons(
            options=["r", "g", "b"],
            disabled=False,
            style={"button_width": "50px", "font_weight": "bold", "font_size": "36px"},
        )
        self.text_input = widgets.Textarea(
            value="",
            placeholder="input your prompt here",
            description="Prompt:",
            disabled=False,
        )
        self.run_button = widgets.Button(
            description="Outpaint",
            tooltip="Run outpainting",
            icon="pen",
            button_style="primary",
        )
        self.export_button = widgets.Button(
            description="Export",
            tooltip="Export the image",
            icon="save",
            button_style="success",
        )
        self.fill_button = widgets.ToggleButtons(
            description="Init mode:",
            options=[
                "edge_pad",
                "patchmatch",
                "cv2_ns",
                "cv2_telea",
                "gaussian",
                "perlin",
            ],
            disabled=False,
            button_style="",
            style={"button_width": "80px", "font_weight": "bold"},
        )

        if self.test_mode:

            def test_button_clicked(btn):
                # lst.append(tuple(base.cursor))
                with self.output:
                    val = self.test_button.value
                    if val == "r":
                        self.fill_selection(
                            np.tile(
                                np.array([255, 0, 0, 255], dtype=np.uint8),
                                (self.selection_size, self.selection_size, 1),
                            )
                        )
                    if val == "g":
                        self.fill_selection(
                            np.tile(
                                np.array([0, 255, 0, 255], dtype=np.uint8),
                                (self.selection_size, self.selection_size, 1),
                            )
                        )
                    if val == "b":
                        self.fill_selection(
                            np.tile(
                                np.array([0, 0, 255, 255], dtype=np.uint8),
                                (self.selection_size, self.selection_size, 1),
                            )
                        )
                    with hold_canvas():
                        self.clear_background()
                        self.draw_buffer()
                        self.draw_selection_box()

            self.run_button.on_click(test_button_clicked)

    def display(self):
        with hold_canvas():
            self.clear_background()
            self.draw_buffer()
            self.draw_selection_box()
        if self.test_mode:
            return [
                self.test_button,
                self.mode_button,
                self.canvas,
                widgets.HBox([self.run_button, self.text_input]),
                self.output,
            ]
        return [
            self.fill_button,
            self.canvas,
            widgets.HBox(
                [self.mode_button, self.run_button, self.export_button, self.text_input]
            ),
            self.output,
        ]

    def clear_background(self):
        # fake transparent background
        h, w, step = self.height, self.width, self.grid_size
        stride = step * 2
        x0, y0 = self.view_pos
        x0 = (-x0) % stride
        y0 = (-y0) % stride
        # self.canvas.clear()
        self.canvas[0].fill_style = "#ffffff"
        self.canvas[0].fill_rect(0, 0, w, h)
        self.canvas[0].fill_style = "#aaaaaa"
        for y in range(y0 - stride, h + step, step):
            start = (x0 - stride) if y // step % 2 == 0 else (x0 - step)
            for x in range(start, w + step, stride):
                self.canvas[0].fill_rect(x, y, step, step)
        self.canvas[0].stroke_rect(0, 0, w, h)

    def update_view_pos(self, xo, yo):
        if abs(xo) + abs(yo) == 0:
            return
        if self.sel_dirty:
            self.write_selection_to_buffer()
        if self.buffer_dirty:
            self.buffer2data()
        self.view_pos[0] -= xo
        self.view_pos[1] -= yo
        self.data2buffer()
        # self.read_selection_from_buffer()

    def update_cursor(self, xo, yo):
        if abs(xo) + abs(yo) == 0:
            return
        if self.sel_dirty:
            self.write_selection_to_buffer()
        self.cursor[0] += xo
        self.cursor[1] += yo
        self.cursor[0] = max(min(self.width - self.selection_size, self.cursor[0]), 0)
        self.cursor[1] = max(min(self.height - self.selection_size, self.cursor[1]), 0)
        # self.read_selection_from_buffer()

    def data2buffer(self):
        x, y = self.view_pos
        h, w = self.height, self.width
        # fill four parts
        for i in range(4):
            pos_src, pos_dst, data = self.select(x, y, i)
            xs0, xs1 = pos_src[0]
            ys0, ys1 = pos_src[1]
            xd0, xd1 = pos_dst[0]
            yd0, yd1 = pos_dst[1]
            self.buffer[yd0:yd1, xd0:xd1, :] = data[ys0:ys1, xs0:xs1, :]

    def buffer2data(self):
        x, y = self.view_pos
        h, w = self.height, self.width
        # fill four parts
        for i in range(4):
            pos_src, pos_dst, data = self.select(x, y, i)
            xs0, xs1 = pos_src[0]
            ys0, ys1 = pos_src[1]
            xd0, xd1 = pos_dst[0]
            yd0, yd1 = pos_dst[1]
            data[ys0:ys1, xs0:xs1, :] = self.buffer[yd0:yd1, xd0:xd1, :]
        self.buffer_dirty = False

    def select(self, x, y, idx):
        w, h = self.width, self.height
        lst = [(0, 0), (0, h), (w, 0), (w, h)]
        if idx == 0:
            x0, y0 = x % self.patch_size, y % self.patch_size
            x1 = min(x0 + w, self.patch_size)
            y1 = min(y0 + h, self.patch_size)
        elif idx == 1:
            y += h
            x0, y0 = x % self.patch_size, y % self.patch_size
            x1 = min(x0 + w, self.patch_size)
            y1 = max(y0 - h, 0)
        elif idx == 2:
            x += w
            x0, y0 = x % self.patch_size, y % self.patch_size
            x1 = max(x0 - w, 0)
            y1 = min(y0 + h, self.patch_size)
        else:
            x += w
            y += h
            x0, y0 = x % self.patch_size, y % self.patch_size
            x1 = max(x0 - w, 0)
            y1 = max(y0 - h, 0)
        xi, yi = x // self.patch_size, y // self.patch_size
        cur = self.data.setdefault(
            (xi, yi), np.zeros((self.patch_size, self.patch_size, 4), dtype=np.uint8)
        )
        x0_img, y0_img = lst[idx]
        x1_img = x0_img + x1 - x0
        y1_img = y0_img + y1 - y0
        sort = lambda a, b: ((a, b) if a < b else (b, a))
        return (
            (sort(x0, x1), sort(y0, y1)),
            (sort(x0_img, x1_img), sort(y0_img, y1_img)),
            cur,
        )

    def draw_buffer(self):
        self.canvas[1].clear()
        self.canvas[1].put_image_data(self.buffer, 0, 0)

    def fill_selection(self, img):
        self.sel_buffer = img
        self.sel_dirty = True

    def draw_selection_box(self):
        x0, y0 = self.cursor
        size = self.selection_size
        if self.sel_dirty:
            self.canvas[2].clear()
            self.canvas[2].put_image_data(self.sel_buffer, x0, y0)
        self.canvas[-1].clear()
        self.canvas[-1].stroke_style = "#0a0a0a"
        self.canvas[-1].stroke_rect(x0, y0, size, size)
        self.canvas[-1].stroke_style = "#ffffff"
        self.canvas[-1].stroke_rect(x0 - 1, y0 - 1, size + 2, size + 2)
        self.canvas[-1].stroke_style = "#000000"
        self.canvas[-1].stroke_rect(x0 - 2, y0 - 2, size + 4, size + 4)

    def write_selection_to_buffer(self):
        x0, y0 = self.cursor
        x1, y1 = x0 + self.selection_size, y0 + self.selection_size
        self.buffer[y0:y1, x0:x1] = self.sel_buffer
        self.sel_dirty = False
        self.sel_buffer = self.sel_buffer_bak.copy()
        self.buffer_dirty = True
        self.buffer_updated = True

    def read_selection_from_buffer(self):
        x0, y0 = self.cursor
        x1, y1 = x0 + self.selection_size, y0 + self.selection_size
        self.sel_buffer = self.buffer[y0:y1, x0:x1]
        self.sel_dirty = False

    def export(self):
        if self.sel_dirty:
            self.write_selection_to_buffer()
        if self.buffer_dirty:
            self.buffer2data()
        xmin, xmax, ymin, ymax = 0, 0, 0, 0
        if len(self.data.keys()) == 0:
            return np.zeros(
                (self.selection_size, self.selection_size, 4), dtype=np.uint8
            )
        for xi, yi in self.data.keys():
            buf = self.data[(xi, yi)]
            if buf.sum() > 0:
                xmin = min(xi, xmin)
                xmax = max(xi, xmax)
                ymin = min(yi, ymin)
                ymax = max(yi, ymax)
        yn = ymax - ymin + 1
        xn = xmax - xmin + 1
        image = np.zeros(
            (yn * self.patch_size, xn * self.patch_size, 4), dtype=np.uint8
        )
        for xi, yi in self.data.keys():
            buf = self.data[(xi, yi)]
            if buf.sum() > 0:
                y0 = (yi - ymin) * self.patch_size
                x0 = (xi - xmin) * self.patch_size
                image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] = buf
        ylst, xlst = image[:, :, -1].nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        image = image[yt : yb + 1, xt : xb + 1]
        return image
