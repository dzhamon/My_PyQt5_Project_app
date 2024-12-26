# Модуль search_in_ListBox быстрого поика элементов в списке
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QListWidget


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.data_list = [
            'C', 'C++', 'Java', 'Python', 'Perl',
            'PHP', 'ASP', 'JS', 'JVC', 'Toyota', 
            'Beatles', 'Deep Purple', 'Middleware', 
            'Compare', 'Komparison', 'Uriah Heep'
        ]

        self.setWindowTitle("Filter List")

        # Layout
        layout = QVBoxLayout()

        # QLineEdit (Entry)
        self.entry = QLineEdit(self)
        self.entry.setPlaceholderText("Start typing...")
        self.entry.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.entry)

        # QListWidget (Listbox)
        self.listbox = QListWidget(self)
        self.update_listbox(self.data_list)
        layout.addWidget(self.listbox)

        self.setLayout(layout)

    def on_text_changed(self, text):
        filtered_data = [item for item in self.data_list if text.lower() in item.lower()]
        self.update_listbox(filtered_data)

    def update_listbox(self, data):
        self.listbox.clear()
        self.listbox.addItems(data)


# Для тестирования
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())
