import cv2 as cv
import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QListWidget, QLabel, QFileDialog, 
                             QInputDialog, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

class OpenCVThread(QThread):
    """OpenCV窗口运行在独立线程中"""
    delete_signal = pyqtSignal()
    save_signal = pyqtSignal()
    update_list_signal = pyqtSignal()  # 新增：用于通知主窗口更新列表
    get_class_id_signal = pyqtSignal()  # 新增：用于获取class_id的信号
    
    def __init__(self, image_path, default_class_id):
        super().__init__()
        self.image_path = image_path
        self.default_class_id = default_class_id  # 默认class_id
        self.running = True
        self.rect_list = []
        self.current_rect = -1
        self.img = None
        self.img_copy = None
        self.clone = None
        self.drawing = False
        self.begin_x = -1
        self.begin_y = -1
        self.end_x = -1
        self.end_y = -1
        self.name = 'YOLO txt maker'
        self.pending_rect_data = None  # 存储待确认的矩形数据
        
    def run(self):
        self.img = cv.imread(self.image_path, 1)
        if self.img is None:
            return
            
        self.img_copy = self.img.copy()
        cv.namedWindow(self.name)
        cv.setMouseCallback(self.name, self.draw_rect)
        self.refresh_image()
        cv.moveWindow(self.name, 100, 100)
        
        while self.running:
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('d') or key == ord('D'):  # D键删除选中的矩形
                self.delete_signal.emit()
            elif key == ord('s') or key == ord('S'):  # S键保存
                self.save_signal.emit()
                
        cv.destroyAllWindows()
    
    def draw_rect(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.begin_x = x
            self.begin_y = y
            self.end_x = x
            self.end_y = y
            self.clone = self.img.copy()

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing:
                self.img = self.clone.copy()
                self.end_x = x
                self.end_y = y
                cv.rectangle(self.img, (self.begin_x, self.begin_y), (self.end_x, self.end_y), (100, 200, 0), 2)
                cv.imshow(self.name, self.img)

        elif event == cv.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x = x
            self.end_y = y
            
            # 检查矩形大小是否合理
            min_size = 10
            if abs(self.end_x - self.begin_x) < min_size or abs(self.end_y - self.begin_y) < min_size:
                self.img = self.clone.copy()
                cv.imshow(self.name, self.img)
                return
                
            cv.rectangle(self.img, (self.begin_x, self.begin_y), (self.end_x, self.end_y), (0, 0, 255), 2)
            cv.imshow(self.name, self.img)
            
            center_x = (self.begin_x + self.end_x) / (2 * self.img.shape[1])
            center_y = (self.begin_y + self.end_y) / (2 * self.img.shape[0])
            width = abs(self.end_x - self.begin_x) / self.img.shape[1]
            height = abs(self.end_y - self.begin_y) / self.img.shape[0]
            
            # 存储矩形数据，等待用户输入class_id
            self.pending_rect_data = [center_x, center_y, width, height]
            self.get_class_id_signal.emit()  # 发送信号请求输入class_id
    
    def add_rect_with_class_id(self, class_id):
        """添加带有指定class_id的矩形"""
        if self.pending_rect_data:
            center_x, center_y, width, height = self.pending_rect_data
            data = [class_id, center_x, center_y, width, height]
            self.rect_list.append(data)
            self.pending_rect_data = None
            self.update_list_signal.emit()  # 发送信号通知主窗口更新列表
    
    def refresh_image(self):
        if self.img_copy is not None:
            self.img = self.img_copy.copy()
            self.draw_exist()
            cv.imshow(self.name, self.img)
    
    def draw_exist(self):
        for i, rect in enumerate(self.rect_list):
            class_id, cx, cy, w, h = rect
            x1 = int((cx - w / 2) * self.img.shape[1])
            y1 = int((cy - h / 2) * self.img.shape[0])
            x2 = int((cx + w / 2) * self.img.shape[1])
            y2 = int((cy + h / 2) * self.img.shape[0])
            color = (0, 0, 255) if i == self.current_rect else (0, 255, 0)
            cv.rectangle(self.img, (x1, y1), (x2, y2), color, 3)
            cv.putText(self.img, f'{class_id}', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def delete_rect(self):
        if self.current_rect >= 0 and self.current_rect < len(self.rect_list):
            self.rect_list.pop(self.current_rect)
            self.current_rect = -1
            self.save_to_file()
            self.refresh_image()
            self.update_list_signal.emit()  # 发送信号通知主窗口更新列表
    
    def save_to_file(self):
        if self.image_path:
            img_name = os.path.splitext(os.path.basename(self.image_path))[0]
            txt_name = f'{img_name}.txt'
            try:
                with open(txt_name, 'w', encoding='utf-8') as f:
                    for rect in self.rect_list:
                        f.write(' '.join(map(str, rect)) + '\n')
                print(f"已保存到: {txt_name}")
            except Exception as e:
                QMessageBox.critical(None, "保存错误", f"保存文件时出错: {str(e)}")
    
    def stop(self):
        self.running = False


class YOLOMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cv_thread = None
        self.image_path = None
        self.class_id = None
        self.init_ui()
        self.load_image()
        
    def init_ui(self):
        self.setWindowTitle('YOLO标注工具')
        self.setGeometry(100, 100, 600, 400)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 说明标签
        info_label = QLabel("操作说明: 鼠标拖拽画矩形(每个矩形可设置不同class_id) | 双击列表编辑class_id | D键删除选中 | S键保存 | ESC退出")
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: blue;")
        layout.addWidget(info_label)
        
        # 矩形列表
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self.edit_rect_class_id)  # 双击编辑class_id
        layout.addWidget(self.list_widget)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        self.delete_btn = QPushButton('删除选中矩形 (D)')
        self.delete_btn.clicked.connect(self.delete_rect)
        self.delete_btn.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold;")
        button_layout.addWidget(self.delete_btn)
        
        self.save_btn = QPushButton('保存标注 (S)')
        self.save_btn.clicked.connect(self.save_to_file)
        self.save_btn.setStyleSheet("background-color: #4ecdc4; color: white; font-weight: bold;")
        button_layout.addWidget(self.save_btn)
        
        self.reload_btn = QPushButton('加载下一张图片')
        self.reload_btn.clicked.connect(self.reload_image)
        self.reload_btn.setStyleSheet("background-color: #45b7d1; color: white; font-weight: bold;")
        button_layout.addWidget(self.reload_btn)
        
        self.quit_btn = QPushButton('退出程序')
        self.quit_btn.clicked.connect(self.quit_program)
        self.quit_btn.setStyleSheet("background-color: #96ceb4; color: white; font-weight: bold;")
        button_layout.addWidget(self.quit_btn)
        
        layout.addLayout(button_layout)
    
    def load_image(self):
        # 选择图片
        self.image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", 
            "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*.*)"
        )
        
        if not self.image_path:
            if QMessageBox.question(self, "退出", "没有选择图片，是否退出程序？") == QMessageBox.Yes:
                self.close()
            else:
                self.load_image()
            return
        
        # 输入class id
        self.class_id, ok = QInputDialog.getText(self, "输入", "请输入class id (0-999):")
        
        if not ok or not self.class_id:
            if QMessageBox.question(self, "退出", "没有输入class id，是否退出程序？") == QMessageBox.Yes:
                self.close()
            else:
                self.load_image()
            return
        
        # 验证class id
        try:
            class_id_int = int(self.class_id)
            if class_id_int < 0 or class_id_int > 999:
                QMessageBox.critical(self, "错误", "class id必须在0-999之间")
                self.load_image()
                return
        except ValueError:
            QMessageBox.critical(self, "错误", "class id必须是数字")
            self.load_image()
            return
        
        # 验证图片
        img = cv.imread(self.image_path, 1)
        if img is None:
            QMessageBox.critical(self, "错误", "无法读取图像文件")
            self.load_image()
            return
        
        # 启动OpenCV线程
        self.start_cv_thread()
        
        # 等待线程启动后再加载标注
        if self.cv_thread:
            self.cv_thread.started.connect(self.load_existing_annotations)
    
    def start_cv_thread(self):
        if self.cv_thread:
            self.cv_thread.stop()
            self.cv_thread.wait()
        
        self.cv_thread = OpenCVThread(self.image_path, self.class_id)
        self.cv_thread.delete_signal.connect(self.delete_rect)
        self.cv_thread.save_signal.connect(self.save_to_file)
        self.cv_thread.update_list_signal.connect(self.update_listbox)  # 连接更新列表信号
        self.cv_thread.get_class_id_signal.connect(self.get_class_id) # 连接获取class_id信号
        self.cv_thread.start()
    
    def get_class_id(self):
        """处理class_id输入对话框"""
        class_id, ok = QInputDialog.getText(
            self, "输入Class ID", 
            f"请输入这个矩形的class id (0-999):\n默认值: {self.class_id}",
            text=self.class_id
        )
        
        if ok and class_id:
            try:
                class_id_int = int(class_id)
                if class_id_int < 0 or class_id_int > 999:
                    QMessageBox.critical(self, "错误", "class id必须在0-999之间")
                    return
                # 添加矩形到列表
                self.cv_thread.add_rect_with_class_id(class_id_int)
            except ValueError:
                QMessageBox.critical(self, "错误", "class id必须是数字")
                return
        else:
            # 用户取消，清除待处理的矩形数据
            self.cv_thread.pending_rect_data = None
    
    def load_existing_annotations(self):
        """加载已有的标注文件"""
        if not self.image_path or not self.cv_thread:
            return
            
        img_name = os.path.splitext(os.path.basename(self.image_path))[0]
        txt_name = f'{img_name}.txt'
        
        if os.path.exists(txt_name):
            try:
                with open(txt_name, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) == 5:
                                class_id, cx, cy, w, h = map(float, parts)
                                # 加载所有标注，不再限制class_id
                                self.cv_thread.rect_list.append([int(class_id), cx, cy, w, h])
                
                # 更新列表显示
                self.update_listbox()
                # 刷新图像显示
                if self.cv_thread:
                    self.cv_thread.refresh_image()
                    
            except Exception as e:
                print(f"加载标注文件时出错: {str(e)}")
    
    def update_listbox(self):
        self.list_widget.clear()
        if self.cv_thread:
            for i, rect in enumerate(self.cv_thread.rect_list):
                class_id, cx, cy, w, h = rect
                self.list_widget.addItem(f'Rect {i}: Class {class_id} [{cx:.3f},{cy:.3f}] {w:.3f}x{h:.3f}')
    
    def on_selection_changed(self):
        current_row = self.list_widget.currentRow()
        if self.cv_thread and current_row >= 0:
            self.cv_thread.current_rect = current_row
            self.cv_thread.refresh_image()
    
    def edit_rect_class_id(self, item):
        """双击编辑矩形的class_id"""
        current_row = self.list_widget.currentRow()
        if self.cv_thread and current_row >= 0 and current_row < len(self.cv_thread.rect_list):
            rect = self.cv_thread.rect_list[current_row]
            current_class_id = rect[0]
            
            new_class_id, ok = QInputDialog.getText(
                self, "编辑Class ID", 
                f"请输入新的class id (0-999):\n当前值: {current_class_id}",
                text=str(current_class_id)
            )
            
            if ok and new_class_id:
                try:
                    new_class_id_int = int(new_class_id)
                    if new_class_id_int < 0 or new_class_id_int > 999:
                        QMessageBox.critical(self, "错误", "class id必须在0-999之间")
                        return
                    # 更新矩形的class_id
                    self.cv_thread.rect_list[current_row][0] = new_class_id_int
                    self.update_listbox()
                    self.cv_thread.refresh_image()
                except ValueError:
                    QMessageBox.critical(self, "错误", "class id必须是数字")
                    return
    
    def delete_rect(self):
        if self.cv_thread:
            self.cv_thread.delete_rect()
            # 不需要手动调用update_listbox()，因为OpenCV线程已经发送了update_list_signal信号
        else:
            QMessageBox.warning(self, "警告", "请先选择一个矩形框")
    
    def save_to_file(self):
        if self.cv_thread:
            self.cv_thread.save_to_file()
    
    def reload_image(self):
        if self.cv_thread:
            self.cv_thread.save_to_file()  # 保存当前标注
            self.cv_thread.stop()
            self.cv_thread.wait()
        
        self.load_image()
    
    def quit_program(self):
        if self.cv_thread:
            self.cv_thread.save_to_file()  # 保存当前标注
            self.cv_thread.stop()
            self.cv_thread.wait()
        self.close()
    
    def closeEvent(self, event):
        self.quit_program()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = YOLOMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()