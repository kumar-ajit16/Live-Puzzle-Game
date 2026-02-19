import cv2
import numpy as np
import random


class PuzzleUI:
    def __init__(self, img, grid_size=3):
        self.img = cv2.resize(img, (600, 600))
        self.grid_size = grid_size

        self.pieces, self.ph, self.pw = self.split_image(self.img, grid_size)
        self.shuffled_pieces, self.order = self.shuffle_pieces(self.pieces)

        self.current_order = self.order.copy()
        self.window = np.zeros_like(self.img)

        self.dragging = False
        self.drag_idx = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.offset = (0, 0)

    def split_image(self, img, grid_size):
        h, w = img.shape[:2]
        ph, pw = h // grid_size, w // grid_size

        pieces = []
        for i in range(grid_size):
            for j in range(grid_size):
                piece = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw].copy()
                pieces.append(piece)

        return pieces, ph, pw

    def shuffle_pieces(self, pieces):
        idx = list(range(len(pieces)))
        random.shuffle(idx)
        shuffled = [pieces[i] for i in idx]
        return shuffled, idx

    def get_piece_index(self, x, y):
        col = x // self.pw
        row = y // self.ph
        idx = row * self.grid_size + col
        if 0 <= idx < len(self.shuffled_pieces):
            return idx
        return None

    def mouse_event(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.get_piece_index(x, y)
            if idx is not None:
                self.dragging = True
                self.drag_idx = idx
                self.offset = (x % self.pw, y % self.ph)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self.mouse_x = x
                self.mouse_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                drop_idx = self.get_piece_index(x, y)
                if drop_idx is not None and drop_idx != self.drag_idx:
                    self.shuffled_pieces[self.drag_idx], self.shuffled_pieces[drop_idx] = \
                        self.shuffled_pieces[drop_idx], self.shuffled_pieces[self.drag_idx]

                    self.current_order[self.drag_idx], self.current_order[drop_idx] = \
                        self.current_order[drop_idx], self.current_order[self.drag_idx]

                self.dragging = False
                self.drag_idx = None

    def draw(self):
        self.window[:] = 255

        border = 2

        for idx, piece in enumerate(self.shuffled_pieces):
            row = idx // self.grid_size
            col = idx % self.grid_size

            y = row * self.ph
            x = col * self.pw

            if self.dragging and idx == self.drag_idx:
                continue

            self.window[y:y+self.ph, x:x+self.pw] = piece
            cv2.rectangle(self.window, (x, y),
                          (x+self.pw, y+self.ph),
                          (200, 200, 200), border)

        if self.dragging and self.drag_idx is not None:
            piece = self.shuffled_pieces[self.drag_idx]

            x0 = self.mouse_x - self.offset[0]
            y0 = self.mouse_y - self.offset[1]

            x1 = x0 + self.pw
            y1 = y0 + self.ph

            x0 = max(0, min(x0, self.window.shape[1] - self.pw))
            y0 = max(0, min(y0, self.window.shape[0] - self.ph))

            self.window[y0:y0+self.ph, x0:x0+self.pw] = piece
            cv2.rectangle(self.window,
                          (x0, y0),
                          (x0+self.pw, y0+self.ph),
                          (0, 255, 255), 3)

    def is_solved(self):
        return self.current_order == list(range(len(self.pieces)))

    def show(self, dark_theme=False, window_name="Puzzle"):
        self.draw()
        cv2.imshow(window_name, self.window)
        return self.is_solved()
