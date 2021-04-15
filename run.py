import sys
import argparse
from ui.draw import ui_edit_widget
from PyQt5 import QtWidgets
import qdarkstyle
import os
import os.path as osp
from artist import Artist

# For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--imgs", type=str, default="./data/imgs/", help='path to img folder')
parser.add_argument("--out", type=str, default="./data/out/", help='path to output')
parser.add_argument("--labels", type=str, default="./data/labels-59.txt", help='out path')
parser.add_argument("--color_ckpt", type=str, default="m./odels/color_weights.pth.tar", help='path of checkpoint for pretrained color model')
parser.add_argument("--seg_ckpt", type=str, default="./models/seg_weights.pth.tar", help='path of checkpoint for pretrained seg model')
parser.add_argument('--cuda', action='store_true', help='enables cuda for better experience')
args = parser.parse_args()

if __name__ == '__main__':
    if not osp.isdir(args.out):
        os.mkdir(args.out)
    _artist = Artist(c_dir=args.color_ckpt, s_dir=args.seg_ckpt, use_cuda=args.cuda)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    edit_widget = QtWidgets.QWidget()
    ui = ui_edit_widget()
    ui.setupUi(edit_widget, args.imgs, _artist, out_dir=args.out, lbls=args.labels)
    edit_widget.show()
    sys.exit(app.exec_())
