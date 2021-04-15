from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import glob
import os
import sys
import copy
import time

_EXTS = ['jpg', 'png']

class ui_edit_widget(object):
    def setupUi(self, edit_widget, image_file, artist, out_dir='./data/out/', lbls="./data/labels-59.txt"):
        self.win_size = 620
        self.result = None
        self.gray_win = None
        self.read_labels(lbls)
        self.vis_seg_map = None
        edit_widget.setObjectName("edit_widget")
        edit_widget.setWindowModality(QtCore.Qt.ApplicationModal)
        edit_widget.resize(1280, 720)

        self.brush_box = QtWidgets.QComboBox(edit_widget)
        self.brush_box.setGeometry(QtCore.QRect(10, 680, 210, 30))
        self.brush_box.setObjectName("brush_box")
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.brush_box.setFont(font)
        self.brush_box.currentIndexChanged.connect(self.brush_box_label_changed)

        self.brush_type = QtWidgets.QLabel(edit_widget)
        self.brush_type.setGeometry(QtCore.QRect(230, 680, 210, 30))
        self.brush_type.setFont(font)
        self.brush_type.setStyleSheet("background-color: rgb(0, 0, 0);\ncolor: rgb(255, 255, 255);")
        self.brush_type.setIndent(10)
        self.brush_type.setObjectName("brush_type")

        self.tab_widget = QtWidgets.QTabWidget(edit_widget)
        self.tab_widget.setGeometry(QtCore.QRect(10, 10, 630, 660))
        self.tab_widget.setObjectName("tab_widget")
        self.drawWidget = GUIDraw(artist, self.palette, self.labels, out_dir=out_dir)
        self.drawWidget.setObjectName("drawWidget")
        self.tab_widget.addTab(self.drawWidget, "")

        self.tab_widget_2 = QtWidgets.QTabWidget(edit_widget)
        self.tab_widget_2.setGeometry(QtCore.QRect(640, 10, 630, 660))
        self.tab_widget_2.setObjectName("tab_widget_2")
        self.visWidget = GUI_VIS()
        self.visWidget.setObjectName("visWidget")
        self.tab_widget_2.addTab(self.visWidget, "")

        self.picker_button = QtWidgets.QPushButton(edit_widget)
        self.picker_button.setGeometry(QtCore.QRect(450, 680, 30, 30))
        self.picker_button.setObjectName("picker_button")
        self.picker_button.setStyleSheet("background-color: white")
        self.picker_button.setIcon(QtGui.QIcon('data/icons/color_picker.png'))
        self.picker_button.setIconSize(QtCore.QSize(30,30))
        self.picker_button.clicked.connect(self.picker_clicked)

        self.skip_button = QtWidgets.QPushButton(edit_widget)
        self.skip_button.setGeometry(QtCore.QRect(880, 680, 90, 30))
        self.skip_button.setObjectName("skip_button")
        self.skip_button.setFont(font)
        self.skip_button.clicked.connect(self.skip_clicked)

        self.save_button = QtWidgets.QPushButton(edit_widget)
        self.save_button.setGeometry(QtCore.QRect(970, 680, 90, 30))
        self.save_button.setObjectName("save_button")
        self.save_button.setFont(font)
        self.save_button.clicked.connect(self.save_clicked)

        self.load_button = QtWidgets.QPushButton(edit_widget)
        self.load_button.setGeometry(QtCore.QRect(1060, 680, 90, 30))
        self.load_button.setObjectName("load_button")
        self.load_button.setFont(font)
        self.load_button.clicked.connect(self.load_clicked)

        self.done_button = QtWidgets.QPushButton(edit_widget)
        self.done_button.setGeometry(QtCore.QRect(1150, 680, 120, 30))
        self.done_button.setObjectName("done_button")
        self.done_button.setFont(font)
        self.done_button.clicked.connect(self.done_clicked)

        self.retranslateUi(edit_widget)
        self.tab_widget.setCurrentIndex(0)
        self.init_brush_box_data()
        QtCore.QMetaObject.connectSlotsByName(edit_widget)

        if '.jpg' in image_file or '.png' in image_file:
            self.drawWidget.init_result(image_file)
        else:
            self.drawWidget.get_batches(image_file)

    def retranslateUi(self, edit_widget):
        _translate = QtCore.QCoreApplication.translate
        edit_widget.setWindowTitle(_translate("edit_widget", "Semantic-driven Colorization"))
        self.brush_type.setText(_translate("edit_widget", "Back Ground"))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.drawWidget), _translate("edit_widget", "Interactive Map"))
        self.tab_widget_2.setTabText(self.tab_widget_2.indexOf(self.visWidget), _translate("edit_widget", "Result"))
        self.save_button.setText(_translate("edit_widget", "Save"))
        self.skip_button.setText(_translate("edit_widget", "Skip"))
        self.load_button.setText(_translate("edit_widget", "Load"))
        self.done_button.setText(_translate("edit_widget", "Save and Next"))
        self.drawWidget.set_vis(self.visWidget)
        self.drawWidget.set_brushbox(self.brush_box, self.brush_type)

    def save_clicked(self):
        print('Save Image')
        self.drawWidget.save_result(_info=str(time.time()).replace('.', '-'))
    
    def load_clicked(self):
        print('Load Image')
        self.drawWidget.load_image()

    def done_clicked(self):
        print('Done')
        self.drawWidget.next_image()
    
    def skip_clicked(self):
        print('Next')
        self.drawWidget.next_image(does_save_result=False)

    def picker_clicked(self):
        print('Picker clicked')
        self.drawWidget.set_picker(not self.drawWidget.picker_activated)
        if self.drawWidget.picker_activated:
            self.picker_button.setStyleSheet("background-color: gray")
        else:
            self.picker_button.setStyleSheet("background-color: white")

    def init_brush_box_data(self):
        self.brush_box.addItems(self.labels)
        for i in range(len(self.labels)):
            _color = self.palette[i]
            self.brush_box.setItemData(i, QtGui.QColor(_color[0], _color[1], _color[2]), QtCore.Qt.BackgroundRole)

    def brush_box_label_changed(self, _index):
        print('Current brush label: ' + str(_index))
        self.drawWidget.update_brush_label()

    def read_labels(self, _dir):
        with open(_dir, 'r') as f:
            labels = f.readlines()
            labels = [k.strip('\n').split(': ')[1] for k in labels]
        self.palette = self.make_palette(len(labels))
        self.labels = labels

    def make_palette(self, num_classes):
        """
        Maps classes to colors in the style of PASCAL VOC.
        Close values are mapped to far colors for segmentation visualization.
        See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
        Takes:
            num_classes: the number of classes
        Gives:
            palette: the colormap as a k x 3 array of RGB colors
        """
        palette = np.zeros((num_classes, 3), dtype=np.uint8)
        for k in range(0, num_classes):
            label = k
            i = 0
            while label:
                palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
                palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
                palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
                label >>= 3
                i += 1
        return palette

class GUIDraw(QtWidgets.QWidget):
    def __init__(self, artist, palette, labels, out_dir='./data/out/', load_size=352, win_size=630):
        QtWidgets.QWidget.__init__(self)
        self.image_file = None
        self.artist = artist
        self.win_size = win_size
        self.load_size = load_size
        self.setFixedSize(win_size, win_size)
        self.move(win_size, win_size)
        self.movie = True
        self.im_gray3 = None
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0
        self.palette = palette
        self.labels = labels
        self.brush_label = 0
        self.setMouseTracking(True)
        self.picker_activated = False
        self.out_dir = out_dir
        self.init_brush()

    def init_result(self, image_file):
        self.read_image(image_file)
        self.reset()

    def init_brush(self):
        self.scale = int(float(self.win_size) / self.load_size)
        print('Brush scale = %f' % self.scale)
        self.brushWidth = 2 * self.scale

    def set_picker(self, _bool):
        self.picker_activated = _bool

    def set_vis(self, visWid):
        self.vis = visWid

    def set_brushbox(self, brush_box, brush_type):
        self.brush_box = brush_box
        self.brush_type = brush_type

    def update_brush_label(self):
        self.brush_label = self.brush_box.currentIndex()
        self.brush_type.setText(self.brush_box.currentText())
        self.brush_type.setStyleSheet("background-color: rgb({}, {}, {});\ncolor: rgb({}, {}, {});".format(
            self.palette[self.brush_label][0],
            self.palette[self.brush_label][1],
            self.palette[self.brush_label][2],
            255 - self.palette[self.brush_label][0],
            255 - self.palette[self.brush_label][1],
            255 - self.palette[self.brush_label][2]
        ))

    def get_batches(self, img_dir):
        self.img_list = [glob.glob(os.path.join(img_dir, '*.' + _ext)) for _ext in _EXTS]
        self.img_list = [k for l in self.img_list for k in l]
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        self.init_result(img_first)

    def next_image(self, does_save_result=True):
        if does_save_result:
            self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('Done.')
            sys.exit()
        img_current = self.img_list[self.image_id]
        self.init_result(img_current)

    def save_result(self, _info='final'):
        path = os.path.abspath(self.image_file)
        path, _ = os.path.splitext(path)
        _name = os.path.basename(self.image_file).split('.')[0]
        save_path = os.path.join(self.out_dir, _name + '_{}'.format(_info))
        print('Saving result to <%s>\n' % save_path)
        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        seg_img = cv2.cvtColor(self.colorize_segmap(self.seg_map) , cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + '_segout.png', seg_img)
        cv2.imwrite(save_path + '_colorout.png', result_bgr)

    def read_image(self, image_file):
        self.image_file = image_file
        im_bgr = cv2.resize(cv2.imread(image_file), (352, 352), interpolation=cv2.INTER_CUBIC)
        self.im_full = im_bgr.copy()
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)

        self.rw = int(round(r * w / 4.0) * 4)
        self.rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (self.rw, self.rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - self.rw) // 2)
        self.dh = int((self.win_size - self.rh) // 2)
        self.win_w = self.rw
        self.win_h = self.rh
        self.im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(self.im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (self.rw, self.rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        self.seg_map = None if self.artist.seg is not None else np.zeros((h,w), dtype=np.uint8)

    def reset(self):
        print('Reset')
        self.result = None
        self.compute_result()
        self.update()

    def scale_point(self, pnt):
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QtCore.QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def load_image(self):
        img_path = QtWidgets.QFileDialog.getOpenFileName(self, 'load an input image')[0]
        print(img_path)
        self.init_result(img_path)

    def compute_result(self):
        color_lab, self.seg_map = self.artist.colorize(self.im_gray[..., np.newaxis], self.seg_map)
        self.vis_seg_map = copy.deepcopy(self.seg_map)
        color_rgb = cv2.cvtColor(color_lab, cv2.COLOR_LAB2RGB)

        self.result = color_rgb
        self.vis.update_result(cv2.resize(color_rgb, (self.rw, self.rh), interpolation=cv2.INTER_CUBIC))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            seg_img = self.colorize_segmap(self.vis_seg_map)
            seg_img = cv2.resize(seg_img, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            im = cv2.addWeighted(im,0.4,seg_img,0.4,0)
            qImg = QtGui.QImage(im.tostring(), im.shape[1], im.shape[0], QtGui.QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)
        painter.end()

    def wheelEvent(self, event):
        d = event.angleDelta().y() / 120
        self.brushWidth = int(min(100.05 * self.scale, max(0, self.brushWidth + d * self.scale)))
        print('Update brushWidth = %f' % self.brushWidth)
        self.draw_segmap(vis=True, pos=self.valid_point(event.pos()))
        self.update()

    def draw_segmap(self, vis=False, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            cent_x = x - self.brushWidth//2 if x > self.brushWidth//2 else 0
            cent_y = y - self.brushWidth//2 if y > self.brushWidth//2 else 0
            self.vis_seg_map = copy.deepcopy(self.seg_map)
            self.vis_seg_map[cent_y:cent_y+self.brushWidth,cent_x:cent_x+self.brushWidth] = self.brush_label

            if not vis:
                self.seg_map = copy.deepcopy(self.vis_seg_map)

    def pick_seg(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            seg_val = self.seg_map[y,x]
            self.brush_box.setCurrentIndex(seg_val)
            self.update_brush_label()
            print('Picker ({},{}): {}'.format(x, y, seg_val))

    def mousePressEvent(self, event):
        print('Mouse pressed', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == QtCore.Qt.LeftButton:
                print('Left clicked')
                if self.picker_activated:
                    self.pick_seg(pos)
                else:
                    self.draw_segmap(vis=False, pos=pos)

    def mouseMoveEvent(self, event):
        pos = self.valid_point(event.pos())
        if pos is not None and not self.picker_activated:
            self.draw_segmap(vis=True, pos=pos)
            if event.buttons() == QtCore.Qt.LeftButton:
                self.draw_segmap(vis=False, pos=pos)
        self.update()

    def mouseReleaseEvent(self, event):
        pos = self.valid_point(event.pos())
        if pos is not None:
            if event.button() == QtCore.Qt.LeftButton:
                if not self.picker_activated:
                    self.compute_result()

    def colorize_segmap(self, seg):
        """
        Replace classes with their colors.
        Takes:
            seg: H x W segmentation image of class IDs
        Gives:
            H x W x 3 image of class colors
        """
        return self.palette[seg.flat].reshape(seg.shape + (3,))

class GUI_VIS(QtWidgets.QWidget):
    def __init__(self, win_size=630, scale=4.0):
        QtWidgets.QWidget.__init__(self)
        self.result = None
        self.win_width = win_size
        self.win_height = win_size
        # self.scale = scale
        self.setFixedSize(self.win_width, self.win_height)

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.result is not None:
            h, w, _ = self.result.shape
            qImg = QtGui.QImage(self.result.tostring(), w, h, QtGui.QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)
        painter.end()

    def update_result(self, result):
        self.result = result
        self.update()

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

