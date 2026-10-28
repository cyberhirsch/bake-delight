"""Small embedded OpenGL OBJ texture viewer used by the GUI."""

import math
import os
import ctypes

# pyopengltk embeds a GLX context in Tk. On Wayland desktops PyOpenGL can
# otherwise select EGL first, which makes pyopengltk fail before the widget is
# created. Tk uses XWayland here, so GLX is the compatible backend.
if os.name != "nt":
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
from PIL import Image

try:
    from OpenGL import GL
    from OpenGL.GL import shaders
    from pyopengltk import OpenGLFrame
except Exception:  # Keep the main application importable when GL is unavailable.
    GL = None
    shaders = None
    OpenGLFrame = object

OPENGL_AVAILABLE = GL is not None and shaders is not None and OpenGLFrame is not object


class OBJMesh:
    """Load only OBJ geometry and UVs. MTL files are intentionally ignored."""

    def __init__(self, path):
        self.path = path
        positions = []
        texcoords = []
        position_indices = []
        uv_indices = []

        # OBJ is a text format, but decoding every line to Unicode is
        # unnecessary for its numeric data. Keeping the parser byte-based
        # also avoids building expanded Python vertex tuples for every face.
        with open(path, "rb") as handle:
            for raw_line in handle:
                parts = raw_line.split()
                if not parts or parts[0].startswith(b"#"):
                    continue
                if parts[0] == b"v" and len(parts) >= 4:
                    positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == b"vt" and len(parts) >= 3:
                    texcoords.append((float(parts[1]), float(parts[2])))
                elif parts[0] == b"f" and len(parts) >= 4:
                    face = []
                    for token in parts[1:]:
                        fields = token.split(b"/")
                        if len(fields) < 2 or not fields[0] or not fields[1]:
                            raise ValueError("The OBJ must provide UV coordinates for every face.")

                        position_index = int(fields[0])
                        uv_index = int(fields[1])
                        position_index = position_index - 1 if position_index > 0 else len(positions) + position_index
                        uv_index = uv_index - 1 if uv_index > 0 else len(texcoords) + uv_index
                        if not (0 <= position_index < len(positions) and 0 <= uv_index < len(texcoords)):
                            raise ValueError("The OBJ contains an out-of-range vertex or UV index.")
                        face.append((position_index, uv_index))

                    # Triangulate polygons as a fan, storing only indices until
                    # all source vertices have been parsed.
                    for index in range(1, len(face) - 1):
                        for position_index, uv_index in (face[0], face[index], face[index + 1]):
                            position_indices.append(position_index)
                            uv_indices.append(uv_index)

        if not position_indices:
            raise ValueError("The OBJ does not contain any faces.")
        if len(uv_indices) != len(position_indices):
            raise ValueError("The OBJ does not contain usable UV coordinates.")

        source_positions = np.asarray(positions, dtype=np.float32)
        source_uvs = np.asarray(texcoords, dtype=np.float32)
        points = source_positions[np.asarray(position_indices, dtype=np.int32)]
        uvs = source_uvs[np.asarray(uv_indices, dtype=np.int32)]
        center = (points.min(axis=0) + points.max(axis=0)) * 0.5
        radius = float(np.max(np.linalg.norm(points - center, axis=1)))
        if radius <= 1e-8:
            raise ValueError("The OBJ has zero-sized geometry.")
        points = (points - center) / radius
        self.vertex_data = np.concatenate((points, uvs), axis=1)
        self.vertex_count = len(position_indices)

    @staticmethod
    def _parse_vertex(token, position_count, uv_count):
        fields = token.split("/")
        if len(fields) < 2 or not fields[1]:
            raise ValueError("The OBJ must provide UV coordinates for every face.")

        def resolve(value, count):
            index = int(value)
            return index - 1 if index > 0 else count + index

        position_index = resolve(fields[0], position_count)
        uv_index = resolve(fields[1], uv_count)
        if not (0 <= position_index < position_count and 0 <= uv_index < uv_count):
            raise ValueError("The OBJ contains an out-of-range vertex or UV index.")
        return position_index, uv_index


_MESH_CACHE = {}


def load_obj(path):
    """Parse an OBJ once per file version; viewports share the result (safe to call from a worker thread)."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    mesh = _MESH_CACHE.get(key)
    if mesh is None:
        mesh = OBJMesh(path)
        while len(_MESH_CACHE) >= 4:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[key] = mesh
    return mesh


class Camera:
    """Orbit camera plus display options that several viewports share, so their views stay linked."""

    def __init__(self):
        self.views = []
        self.fov = 50.0
        self.wireframe = False
        self.wire_color = (0.92, 0.92, 0.92, 0.45)
        self.checker = False         # procedural UV checker instead of the texture
        self.checker_cells = 32
        self.texture_filter = "mipmap"  # "mipmap", "linear" or "nearest"
        self.background = (0.04, 0.04, 0.04)
        self.orbit_speed = 0.5       # degrees per pixel dragged
        self.zoom_step = 1.14
        self.reset()

    def reset(self):
        self.yaw, self.pitch, self.distance = 25.0, 15.0, 3.0

    def changed(self):
        for view in self.views:
            view.redraw()


class _Slot:
    """One mesh + texture pair. Paths are what the GUI asked for; the GL objects belong to one context."""

    def __init__(self):
        self.mesh = None
        self.image = None          # (width, height, RGBA bytes) awaiting upload
        self.texture_path = None
        self.vertex_buffer = None
        self.uploaded_mesh = None
        self.texture_id = None
        self.uploaded_texture = None
        self.has_mipmaps = False

    @property
    def triangles(self):
        return self.mesh.vertex_count // 3 if self.mesh is not None else 0


class TextureViewport(OpenGLFrame):
    """Unlit, UV-textured OBJ viewport with orbit and zoom controls.

    It holds two slots, "a" (before) and "b" (after), and shows one of them or
    both split by a draggable wipe line ("wipe"). Viewports constructed with
    the same Camera orbit and zoom together.
    """

    VERTEX_SHADER = """
        #version 120
        attribute vec3 position;
        attribute vec2 texcoord;
        uniform mat4 transform;
        varying vec2 uv;
        void main() { uv = texcoord; gl_Position = transform * vec4(position, 1.0); }
    """
    FRAGMENT_SHADER = """
        #version 120
        uniform sampler2D texture_map;
        uniform int mode;            // 0 texture, 1 flat colour (wireframe), 2 UV checker
        uniform vec4 flat_color;
        uniform float cells;
        varying vec2 uv;
        void main() {
            if (mode == 1) { gl_FragColor = flat_color; return; }
            if (mode == 2) {
                float c = mod(floor(uv.x * cells) + floor(uv.y * cells), 2.0);
                gl_FragColor = vec4(mix(vec3(0.16), vec3(0.82), c), 1.0);
                return;
            }
            gl_FragColor = texture2D(texture_map, uv);
        }
    """

    def __init__(self, master, camera=None, **kwargs):
        if GL is None:
            raise RuntimeError("OpenGL preview dependencies are not installed.")
        super().__init__(master, **kwargs)
        self.program = None
        self.camera = camera or Camera()
        self.camera.views.append(self)
        self.slots = {"a": _Slot(), "b": _Slot()}
        self.mode = "b"            # "a", "b" or "wipe"
        self.split = 0.5           # wipe position, fraction of the width from the left
        self._drag_start = None
        self._dragging_split = False
        self.bind("<ButtonPress-1>", self._begin_drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._end_drag)
        self.bind("<Motion>", self._hover)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda event: self._zoom(-1))
        self.bind("<Button-5>", lambda event: self._zoom(1))

    def initgl(self):
        GL.glClearColor(0.04, 0.04, 0.04, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        self.program = shaders.compileProgram(
            shaders.compileShader(self.VERTEX_SHADER, GL.GL_VERTEX_SHADER),
            shaders.compileShader(self.FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
        )

    def tkResize(self, event):
        # The base class re-runs initgl on every resize, in whatever context is
        # current; with several viewports that compiles into the wrong one.
        self.width, self.height = event.width, event.height
        self.redraw()

    def _display(self):
        self.redraw()

    # ------------------------------------------------------------ content

    def set_slot(self, slot, mesh_path=None, texture_path=None):
        """Show `mesh_path` with `texture_path` in slot "a" or "b". Parsing and decoding happen
        here, so errors reach the caller; the GL upload waits until the context exists."""
        target = self.slots[slot]
        target.mesh = load_obj(mesh_path) if mesh_path and os.path.isfile(mesh_path) else None
        if texture_path and os.path.isfile(texture_path):
            stamp = (texture_path, os.path.getmtime(texture_path))
            if stamp != target.texture_path:
                with Image.open(texture_path) as source:
                    image = source.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    target.image = (image.width, image.height, image.tobytes())
                target.texture_path = stamp
        else:
            target.image, target.texture_path = None, None
        self.redraw()

    def set_mode(self, mode):
        self.mode = mode
        self.configure(cursor="")
        self.redraw()

    def _sync(self, slot):
        """Upload whatever changed since the last draw, in this viewport's context."""
        if slot.mesh is not slot.uploaded_mesh:
            if slot.mesh is not None:
                if slot.vertex_buffer is None:
                    slot.vertex_buffer = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, slot.vertex_buffer)
                GL.glBufferData(GL.GL_ARRAY_BUFFER, slot.mesh.vertex_data.nbytes, slot.mesh.vertex_data, GL.GL_STATIC_DRAW)
            slot.uploaded_mesh = slot.mesh
        if slot.image is not None:
            width, height, pixels = slot.image
            if slot.texture_id is None:
                slot.texture_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, slot.texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
            try:
                GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
                slot.has_mipmaps = True
            except Exception:  # pre-GL3 driver: no mipmaps, linear filtering only
                slot.has_mipmaps = False
            slot.uploaded_texture, slot.image = slot.texture_path, None
        elif slot.texture_path is None:
            slot.uploaded_texture = None

    # ------------------------------------------------------------ camera

    def reset_view(self):
        self.camera.reset()
        self.camera.changed()

    def set_fov(self, fov):
        self.camera.fov = max(20.0, min(120.0, float(fov)))
        self.camera.changed()

    def _near_split(self, x):
        return self.mode == "wipe" and abs(x - self.split * max(1, self.winfo_width())) <= 6

    def _hover(self, event):
        self.configure(cursor="sb_h_double_arrow" if self._near_split(event.x) else "")

    def _begin_drag(self, event):
        self._dragging_split = self._near_split(event.x)
        self._drag_start = (event.x, event.y, self.camera.yaw, self.camera.pitch)

    def _drag(self, event):
        if self._dragging_split:
            self.split = max(0.0, min(1.0, event.x / max(1, self.winfo_width())))
            self.redraw()
        elif self._drag_start:
            x, y, yaw, pitch = self._drag_start
            speed = self.camera.orbit_speed
            self.camera.yaw = yaw - (event.x - x) * speed
            self.camera.pitch = max(-89.0, min(89.0, pitch + (event.y - y) * speed))
            self.camera.changed()

    def _end_drag(self, _event):
        self._drag_start = None
        self._dragging_split = False

    def _wheel(self, event):
        self._zoom(-1 if event.delta > 0 else 1)

    def _zoom(self, direction):
        camera = self.camera
        camera.distance = max(1.2, min(12.0, camera.distance * (1.0 / camera.zoom_step if direction < 0 else camera.zoom_step)))
        camera.changed()

    # ------------------------------------------------------------ drawing

    def redraw(self):
        if not self.winfo_exists() or not self.program or not self.winfo_ismapped():
            return
        self.tkMakeCurrent()  # several viewports means several GL contexts
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        GL.glViewport(0, 0, width, height)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(*self.camera.background, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        for slot in self.slots.values():
            self._sync(slot)

        camera = self.camera
        projection = self._perspective(math.radians(camera.fov), width / height, 0.1, 100.0)
        yaw, pitch = math.radians(camera.yaw), math.radians(camera.pitch)
        eye = np.array([
            camera.distance * math.cos(pitch) * math.sin(yaw),
            camera.distance * math.sin(pitch),
            camera.distance * math.cos(pitch) * math.cos(yaw),
        ], dtype=np.float32)
        view = self._look_at(eye, np.zeros(3, dtype=np.float32), np.array([0, 1, 0], dtype=np.float32))
        transform = projection @ view

        if self.mode == "wipe":
            split = int(round(self.split * width))
            GL.glEnable(GL.GL_SCISSOR_TEST)
            GL.glScissor(0, 0, split, height)
            self._draw(self.slots["a"], transform)
            GL.glScissor(split, 0, width - split, height)
            self._draw(self.slots["b"], transform)
            GL.glScissor(max(0, split - 1), 0, 2, height)
            GL.glClearColor(0.88, 0.88, 0.88, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glDisable(GL.GL_SCISSOR_TEST)
        else:
            self._draw(self.slots[self.mode], transform)
        self.tkSwapBuffers()

    def _draw(self, slot, transform):
        camera = self.camera
        if slot.uploaded_mesh is None or (slot.uploaded_texture is None and not camera.checker):
            return
        program = self.program
        GL.glUseProgram(program)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, "transform"), 1, GL.GL_TRUE, transform)
        GL.glUniform1i(GL.glGetUniformLocation(program, "mode"), 2 if camera.checker else 0)
        GL.glUniform1f(GL.glGetUniformLocation(program, "cells"), float(camera.checker_cells))
        if slot.uploaded_texture is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, slot.texture_id)
            nearest = camera.texture_filter == "nearest"
            mipmap = camera.texture_filter == "mipmap" and slot.has_mipmaps
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST if nearest else GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                               GL.GL_NEAREST if nearest else GL.GL_LINEAR_MIPMAP_LINEAR if mipmap else GL.GL_LINEAR)
            GL.glUniform1i(GL.glGetUniformLocation(program, "texture_map"), 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, slot.vertex_buffer)
        stride = 5 * 4
        position = GL.glGetAttribLocation(self.program, "position")
        texcoord = GL.glGetAttribLocation(self.program, "texcoord")
        GL.glEnableVertexAttribArray(position)
        GL.glVertexAttribPointer(position, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(texcoord)
        GL.glVertexAttribPointer(texcoord, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, slot.uploaded_mesh.vertex_count)
        if camera.wireframe:
            # Second pass over the same triangles; OBJ polygons are fan-triangulated, so quads show a diagonal.
            GL.glUniform1i(GL.glGetUniformLocation(program, "mode"), 1)
            GL.glUniform4f(GL.glGetUniformLocation(program, "flat_color"), *camera.wire_color)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, slot.uploaded_mesh.vertex_count)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glDisable(GL.GL_BLEND)
        GL.glDisableVertexAttribArray(position)
        GL.glDisableVertexAttribArray(texcoord)
        GL.glUseProgram(0)

    @staticmethod
    def _perspective(fov, aspect, near, far):
        f = 1.0 / math.tan(fov / 2.0)
        result = np.zeros((4, 4), dtype=np.float32)
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = (2 * far * near) / (near - far)
        result[3, 2] = -1.0
        return result

    @staticmethod
    def _look_at(eye, target, up):
        forward = target - eye
        forward /= np.linalg.norm(forward)
        side = np.cross(forward, up)
        side /= np.linalg.norm(side)
        true_up = np.cross(side, forward)
        result = np.eye(4, dtype=np.float32)
        result[0, :3], result[1, :3], result[2, :3] = side, true_up, -forward
        result[0, 3], result[1, 3], result[2, 3] = -np.dot(side, eye), -np.dot(true_up, eye), np.dot(forward, eye)
        return result
