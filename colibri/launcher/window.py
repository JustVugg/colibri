"""Native Qt desktop interface for launching an existing Colibri install."""

from __future__ import annotations

import dataclasses
import time
import webbrowser
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool, QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .backend import build_launch, command_preview, inspect_model
from .domain import LauncherError, LaunchOptions, Preflight, ProcessEvent
from .installation import find_installation
from .settings import load_settings, save_settings
from .supervisor import Supervisor
from .theme import apply_theme, stylesheet
from .workers import FunctionWorker, ProcessEventBridge


class LauncherWindow(QMainWindow):
    """Friendly one-model launcher backed by the adapter and Supervisor."""

    def __init__(self, settings_path: Path | None = None):
        super().__init__()
        self.settings_path = settings_path
        self._settings, warning = load_settings(settings_path)
        apply_theme(QApplication.instance(), self._settings.get("theme", "light"))
        self._installation = None
        self._installation_generation = 0
        self._python_edit_pending = False
        self._preflight: Preflight | None = None
        self._preflight_key: tuple[str, LaunchOptions] | None = None
        self._pending_start = False
        self._launch_in_progress = False
        self._launch_generation = 0
        self._launch_request: tuple[int, str, Any, LaunchOptions] | None = None
        self._supervisor: Supervisor | None = None
        self._supervisor_start_pending = False
        self._stop_requested = False
        self._requested_compute = "auto"
        self._cuda_fallback_observed = False
        # App-owned threads may outlive a closing window briefly. Tokens and
        # the alive flag make their eventual signals harmless without making
        # close wait on an external diagnostic timeout.
        self._workers = QThreadPool.globalInstance()
        self._request_number = 0
        self._latest: dict[str, str] = {}
        self._callbacks: dict[str, tuple[Callable[[Any], None], Callable[[str], None]]] = {}
        self._alive = True
        self._allow_close = False
        self._close_after_stop = False
        self._updating_options = False
        self._log_lines: deque[str] = deque(maxlen=1000)
        self._load_started = 0.0
        self._chat_opened = False
        self._runtime_state = "stopped"

        self._event_bridge = ProcessEventBridge(self)
        self._event_bridge.received.connect(self._on_process_event)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._preflight_timer = QTimer(self)
        self._preflight_timer.setSingleShot(True)
        self._preflight_timer.setInterval(120)
        self._preflight_timer.timeout.connect(self._begin_preflight)

        self._build_ui()
        self._populate_library()
        if warning:
            self.readiness_label.setText(warning)
            self.readiness_label.setProperty("tone", "warning")
        self._discover_installation()

    def _build_ui(self) -> None:
        self.setWindowTitle("Colibri Launcher")
        self.resize(1040, 720)
        self.setMinimumSize(720, 560)
        self.setStyleSheet(stylesheet(self._settings.get("theme", "light")))

        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(root)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(260)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(22, 24, 18, 20)
        side_layout.setSpacing(12)
        brand = QLabel("COLIBRI")
        brand.setObjectName("brand")
        brand.setAccessibleName("Colibri Launcher")
        side_layout.addWidget(brand)
        subtitle = QLabel("Local model library")
        subtitle.setObjectName("muted")
        side_layout.addWidget(subtitle)

        self.model_list = QListWidget()
        self.model_list.setObjectName("modelList")
        self.model_list.setAccessibleName("Saved models")
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_list.currentItemChanged.connect(self._selection_changed)
        side_layout.addWidget(self.model_list, 1)

        self.add_model_button = QPushButton("＋  Add model folder")
        self.add_model_button.setObjectName("addModelButton")
        self.add_model_button.setAccessibleName("Add a model folder")
        self.add_model_button.clicked.connect(self._choose_model)
        side_layout.addWidget(self.add_model_button)
        library_actions = QHBoxLayout()
        self.rename_model_button = QPushButton("Rename")
        self.rename_model_button.setAccessibleName("Rename selected model")
        self.rename_model_button.clicked.connect(self._rename_model)
        self.remove_model_button = QPushButton("Remove")
        self.remove_model_button.setAccessibleName("Remove selected model from library")
        self.remove_model_button.clicked.connect(self._remove_model)
        library_actions.addWidget(self.rename_model_button)
        library_actions.addWidget(self.remove_model_button)
        side_layout.addLayout(library_actions)

        theme_row = QHBoxLayout()
        theme_label = QLabel("Theme")
        theme_label.setObjectName("muted")
        self.theme_combo = QComboBox()
        self.theme_combo.setAccessibleName("Launcher theme")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(self._settings.get("theme", "light")))
        theme_label.setBuddy(self.theme_combo)
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        theme_row.addWidget(theme_label)
        theme_row.addWidget(self.theme_combo, 1)
        side_layout.addSpacing(8)
        side_layout.addLayout(theme_row)
        outer.addWidget(sidebar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(38, 30, 42, 34)
        content.setSpacing(16)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

        heading = QLabel("Launch a local model")
        heading.setObjectName("heading")
        content.addWidget(heading)
        self.model_path_label = QLabel("Choose a saved model to begin")
        self.model_path_label.setObjectName("pathLabel")
        self.model_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_path_label.setWordWrap(True)
        content.addWidget(self.model_path_label)

        self.settings_warning_label = QLabel()
        self.settings_warning_label.setAccessibleName("Settings could not be saved")
        self.settings_warning_label.setWordWrap(True)
        self.settings_warning_label.hide()
        content.addWidget(self.settings_warning_label)

        self.empty_state_card = QFrame()
        self.empty_state_card.setObjectName("emptyState")
        empty_layout = QVBoxLayout(self.empty_state_card)
        empty_layout.setContentsMargins(20, 18, 20, 18)
        self.empty_state = QLabel(
            "Add a model folder to create your library. Colibri will inspect the folder and show what is ready."
        )
        self.empty_state.setWordWrap(True)
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(self.empty_state)
        self.empty_add_button = QPushButton("Add a model folder")
        self.empty_add_button.setAccessibleName("Add your first model folder")
        self.empty_add_button.clicked.connect(self._choose_model)
        empty_layout.addWidget(self.empty_add_button, 0, Qt.AlignmentFlag.AlignCenter)
        content.addWidget(self.empty_state_card)

        self.controls_frame = QFrame()
        self.controls_frame.setObjectName("card")
        controls = QFormLayout(self.controls_frame)
        controls.setContentsMargins(22, 20, 22, 20)
        controls.setHorizontalSpacing(24)
        controls.setVerticalSpacing(14)
        self.mode_combo = QComboBox()
        self.mode_combo.setAccessibleName("Application mode")
        self.mode_combo.addItem("Web Chat", "web")
        self.mode_combo.addItem("API Server", "serve")
        self.compute_combo = QComboBox()
        self.compute_combo.setAccessibleName("Compute choice")
        self.compute_combo.addItem("Automatic", "auto")
        self.compute_combo.addItem("CPU only", "cpu")
        self.compute_combo.addItem("NVIDIA CUDA", "cuda")
        self.gpu_combo = QComboBox()
        self.gpu_combo.setAccessibleName("NVIDIA GPU devices")
        self.gpu_combo.addItem("All compatible GPUs (automatic)", ())
        compute_box = QWidget()
        compute_layout = QVBoxLayout(compute_box)
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(4)
        compute_layout.addWidget(self.compute_combo)
        self.compute_reason = QLabel("Hardware compatibility will appear after inspection.")
        self.compute_reason.setObjectName("helpText")
        self.compute_reason.setWordWrap(True)
        compute_layout.addWidget(self.compute_reason)
        controls.addRow("App", self.mode_combo)
        controls.addRow("Compute", compute_box)
        controls.addRow("GPU devices", self.gpu_combo)
        content.addWidget(self.controls_frame)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced settings")
        self.advanced_toggle.setAccessibleName("Show advanced settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        content.addWidget(self.advanced_toggle)
        self.advanced_panel = QFrame()
        self.advanced_panel.setObjectName("card")
        advanced = QFormLayout(self.advanced_panel)
        advanced.setContentsMargins(22, 18, 22, 18)
        advanced.setSpacing(12)
        self.ram_spin = self._automatic_spin("RAM budget", 0, 2048, " GB")
        self.vram_spin = QDoubleSpinBox()
        self.vram_spin.setAccessibleName("GPU memory budget")
        self.vram_spin.setRange(0.0, 512.0)
        self.vram_spin.setSuffix(" GB")
        self.vram_spin.setSpecialValueText("Automatic")
        self.context_spin = self._automatic_spin("Context length", 0, 1_048_576, " tokens")
        self.max_tokens_spin = self._automatic_spin("Response limit", 0, 1_048_576, " tokens")
        self.port_spin = QSpinBox()
        self.port_spin.setAccessibleName("Server port")
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8000)
        advanced.addRow("RAM budget", self.ram_spin)
        advanced.addRow("GPU memory", self.vram_spin)
        advanced.addRow("Context", self.context_spin)
        advanced.addRow("Response limit", self.max_tokens_spin)
        advanced.addRow("Local port", self.port_spin)
        self.advanced_panel.hide()
        content.addWidget(self.advanced_panel)

        readiness_card = QFrame()
        readiness_card.setObjectName("card")
        ready_layout = QVBoxLayout(readiness_card)
        ready_layout.setContentsMargins(22, 18, 22, 18)
        ready_title = QLabel("Readiness")
        ready_title.setObjectName("cardTitle")
        ready_layout.addWidget(ready_title)
        self.readiness_label = QLabel("Choose a model to check readiness.")
        self.readiness_label.setObjectName("readinessLabel")
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        ready_layout.addWidget(self.readiness_label)
        self.retry_button = QPushButton("Retry check")
        self.retry_button.setAccessibleName("Retry model readiness check")
        self.retry_button.clicked.connect(self._retry)
        self.retry_button.hide()
        ready_layout.addWidget(self.retry_button, 0, Qt.AlignmentFlag.AlignLeft)
        content.addWidget(readiness_card)

        action_row = QHBoxLayout()
        state_box = QVBoxLayout()
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("statusLabel")
        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("helpText")
        state_box.addWidget(self.status_label)
        state_box.addWidget(self.elapsed_label)
        action_row.addLayout(state_box)
        action_row.addStretch(1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setAccessibleName("Stop the running model")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setAccessibleName("Start selected model")
        self.start_button.setDefault(True)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.start_button)
        content.addLayout(action_row)

        self.api_address = QLineEdit()
        self.api_address.setReadOnly(True)
        self.api_address.setAccessibleName("API address and model identifier")
        self.api_address.hide()
        content.addWidget(self.api_address)

        self.details_toggle = QToolButton()
        self.details_toggle.setText("Details and logs")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.toggled.connect(self._toggle_details)
        content.addWidget(self.details_toggle)
        self.details_panel = QFrame()
        self.details_panel.setObjectName("card")
        details = QVBoxLayout(self.details_panel)
        details.setContentsMargins(18, 16, 18, 16)
        self.command_preview_edit = QLineEdit()
        self.command_preview_edit.setReadOnly(True)
        self.command_preview_edit.setPlaceholderText("Launch command appears after a successful check")
        self.command_preview_edit.setAccessibleName("Launch command preview")
        details.addWidget(self.command_preview_edit)
        detail_actions = QHBoxLayout()
        self.copy_command_button = QPushButton("Copy command")
        self.copy_command_button.clicked.connect(self._copy_command)
        self.export_logs_button = QPushButton("Export logs…")
        self.export_logs_button.clicked.connect(self._export_logs)
        detail_actions.addWidget(self.copy_command_button)
        detail_actions.addWidget(self.export_logs_button)
        detail_actions.addStretch(1)
        details.addLayout(detail_actions)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setAccessibleName("Launcher logs")
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setMinimumHeight(150)
        details.addWidget(self.log_view)
        self.details_panel.hide()
        content.addWidget(self.details_panel)

        settings_toggle = QToolButton()
        settings_toggle.setText("Python interpreter")
        settings_toggle.setCheckable(True)
        settings_toggle.setArrowType(Qt.ArrowType.RightArrow)
        settings_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        content.addWidget(settings_toggle)
        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("card")
        settings_form = QFormLayout(self.settings_panel)
        self.python_edit = QLineEdit(str(self._settings.get("python", "")))
        self.python_edit.setAccessibleName("Python executable")
        self.python_edit.textEdited.connect(self._python_edited)
        self.python_edit.editingFinished.connect(self._commit_python_edit)
        self.python_browse_button = QPushButton("Browse…")
        self.python_browse_button.clicked.connect(self._browse_python)
        python_row = self._browse_row(self.python_edit, self.python_browse_button)
        settings_form.addRow("Python", python_row)
        self.settings_panel.hide()
        settings_toggle.toggled.connect(lambda shown: self._toggle_panel(settings_toggle, self.settings_panel, shown))
        content.addWidget(self.settings_panel)
        content.addStretch(1)

        for widget in (
            self.mode_combo,
            self.compute_combo,
            self.gpu_combo,
            self.ram_spin,
            self.vram_spin,
            self.context_spin,
            self.max_tokens_spin,
            self.port_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._options_changed)
            else:
                widget.valueChanged.connect(self._options_changed)

    @staticmethod
    def _browse_row(edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row

    @staticmethod
    def _automatic_spin(name: str, minimum: int, maximum: int, suffix: str) -> QSpinBox:
        spin = QSpinBox()
        spin.setAccessibleName(name)
        spin.setRange(minimum, maximum)
        spin.setSuffix(suffix)
        spin.setSpecialValueText("Automatic")
        return spin

    def _populate_library(self) -> None:
        self.model_list.clear()
        selected = str(self._settings.get("selected_model", ""))
        selected_item = None
        for entry in self._settings.get("models", []):
            item = QListWidgetItem(str(entry.get("name") or Path(entry["path"]).name))
            item.setData(Qt.ItemDataRole.UserRole, entry["path"])
            item.setToolTip(entry["path"])
            self.model_list.addItem(item)
            if entry["path"] == selected:
                selected_item = item
        if selected_item is not None:
            self.model_list.setCurrentItem(selected_item)
        elif self.model_list.count():
            self.model_list.setCurrentRow(0)
        self._update_empty_state()

    def _invalidate_installation(self) -> None:
        self._invalidate_pending_launch()
        self._installation_generation += 1
        self._installation = None
        self._preflight = None
        self._preflight_key = None
        self._pending_start = False
        self._preflight_timer.stop()
        self._invalidate_worker_category("installation")
        self._invalidate_worker_category("preflight")
        self.start_button.setEnabled(False)
        self.retry_button.hide()

    @Slot(str)
    def _python_edited(self, _text: str) -> None:
        self._python_edit_pending = True
        self._invalidate_installation()
        self.readiness_label.setText("Python changed. Finish editing to check the interpreter.")

    @Slot()
    def _commit_python_edit(self) -> None:
        if self._alive and self._python_edit_pending:
            self._discover_installation()

    def _discover_installation(self) -> None:
        self._python_edit_pending = False
        self._invalidate_installation()
        python = self.python_edit.text().strip() or None
        generation = self._installation_generation
        self.readiness_label.setText("Looking for Colibri…")
        self._run(
            "installation",
            find_installation,
            None,
            python,
            on_result=lambda installation: self._installation_ready(generation, installation),
            on_error=lambda message: self._installation_failed(generation, message),
        )

    def _installation_ready(self, generation: int, installation: Any) -> None:
        if generation != self._installation_generation:
            return
        self._installation = installation
        self.python_edit.setText(str(installation.python))
        self._settings["installation"] = str(installation.root)
        self._settings["python"] = str(installation.python)
        self._save()
        if self.model_list.currentItem() is not None:
            self._begin_preflight()
        else:
            self.readiness_label.setText("Colibri found. Add a model folder when you are ready.")

    def _installation_failed(self, generation: int, message: str) -> None:
        if generation != self._installation_generation:
            return
        self._installation = None
        self.start_button.setEnabled(False)
        self.readiness_label.setText(message)
        self.retry_button.show()

    def _choose_model(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a model folder")
        if not folder:
            return
        normalized = str(Path(folder).resolve())
        for row in range(self.model_list.count()):
            if self.model_list.item(row).data(Qt.ItemDataRole.UserRole) == normalized:
                self.model_list.setCurrentRow(row)
                return
        entry = {"path": normalized, "name": Path(normalized).name, "options": dataclasses.asdict(LaunchOptions())}
        self._settings.setdefault("models", []).append(entry)
        self._settings["selected_model"] = normalized
        item = QListWidgetItem(entry["name"])
        item.setData(Qt.ItemDataRole.UserRole, normalized)
        item.setToolTip(normalized)
        self.model_list.addItem(item)
        self.model_list.setCurrentItem(item)
        self._save()
        self._update_empty_state()

    def _rename_model(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename model", "Name", text=item.text())
        name = name.strip()
        if not accepted or not name:
            return
        item.setText(name)
        entry = self._selected_entry()
        if entry is not None:
            entry["name"] = name
            self._save()

    def _remove_model(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            return
        answer = QMessageBox.question(
            self,
            "Remove from library?",
            "This removes the shortcut only. The model folder stays on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self._settings["models"] = [entry for entry in self._settings.get("models", []) if entry["path"] != path]
        row = self.model_list.row(item)
        self.model_list.takeItem(row)
        self._settings["selected_model"] = (
            self.model_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.model_list.currentItem() else ""
        )
        self._save()
        self._preflight = None
        self._update_empty_state()
        if not self.model_list.count():
            self.model_path_label.setText("Choose a saved model to begin")
            self.readiness_label.setText("Choose a model to check readiness.")
            self.start_button.setEnabled(False)

    def _selection_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self._invalidate_pending_launch()
        self._preflight_timer.stop()
        self._preflight = None
        self._preflight_key = None
        self._pending_start = False
        if current is None:
            self._update_empty_state()
            return
        path = str(current.data(Qt.ItemDataRole.UserRole))
        self._settings["selected_model"] = path
        self.model_path_label.setText(path)
        self._updating_options = True
        try:
            self.context_spin.setMaximum(1_048_576)
            self.max_tokens_spin.setMaximum(1_048_576)
        finally:
            self._updating_options = False
        self._apply_options(self._selected_entry().get("options", {}) if self._selected_entry() else {})
        self._save()
        self.readiness_label.setText("Checking this model…")
        self.start_button.setEnabled(False)
        self.retry_button.hide()
        if self._installation is not None:
            self._begin_preflight()

    def _selected_entry(self) -> dict[str, Any] | None:
        item = self.model_list.currentItem()
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return next((entry for entry in self._settings.get("models", []) if entry["path"] == path), None)

    def _apply_options(self, values: dict[str, Any]) -> None:
        defaults = dataclasses.asdict(LaunchOptions())
        defaults.update({key: value for key, value in values.items() if key in defaults})
        self._updating_options = True
        try:
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(defaults["mode"])))
            self.compute_combo.setCurrentIndex(max(0, self.compute_combo.findData(defaults["compute"])))
            self._set_gpu_choices((), tuple(defaults["gpu_ids"]))
            self.ram_spin.setValue(defaults["ram_gb"])
            self.vram_spin.setValue(defaults["vram_gb"])
            self.context_spin.setValue(defaults["context"])
            self.max_tokens_spin.setValue(defaults["max_tokens"])
            self.port_spin.setValue(defaults["port"])
        finally:
            self._updating_options = False
        self._update_gpu_control_enabled()

    @staticmethod
    def _gpu_ids_from_data(value: Any) -> tuple[int, ...]:
        if not isinstance(value, (tuple, list)):
            return ()
        return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))

    def _set_gpu_choices(self, gpus: tuple[Any, ...], selected_ids: tuple[int, ...]) -> None:
        selected_ids = tuple(selected_ids)
        self.gpu_combo.clear()
        self.gpu_combo.addItem("All compatible GPUs (automatic)", ())
        known = {gpu.index: gpu for gpu in gpus}
        selected_index = 0 if not selected_ids else -1
        for gpu in sorted(gpus, key=lambda item: item.index):
            memory = f" · {gpu.memory_gb:g} GB" if gpu.memory_gb else ""
            self.gpu_combo.addItem(f"GPU {gpu.index}: {gpu.name}{memory}", (gpu.index,))
            if selected_ids == (gpu.index,):
                selected_index = self.gpu_combo.count() - 1
        if selected_ids and selected_index < 0:
            missing = [value for value in selected_ids if value not in known]
            numbers = ", ".join(str(value) for value in selected_ids)
            if missing:
                label = f"Saved GPUs {numbers} (not detected)"
            else:
                names = " + ".join(f"GPU {value}: {known[value].name}" for value in selected_ids)
                label = f"Saved group: {names}"
            self.gpu_combo.addItem(label, selected_ids)
            selected_index = self.gpu_combo.count() - 1
        self.gpu_combo.setCurrentIndex(selected_index)

    def _update_gpu_control_enabled(self) -> None:
        self.gpu_combo.setEnabled(not self._launch_in_progress and self.compute_combo.currentData() != "cpu")

    def _options(self) -> LaunchOptions:
        return LaunchOptions(
            mode=str(self.mode_combo.currentData()),
            compute=str(self.compute_combo.currentData()),
            gpu_ids=self._gpu_ids_from_data(self.gpu_combo.currentData()),
            ram_gb=self.ram_spin.value(),
            vram_gb=self.vram_spin.value(),
            context=self.context_spin.value(),
            max_tokens=self.max_tokens_spin.value(),
            port=self.port_spin.value(),
        )

    def _set_launch_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.model_list,
            self.add_model_button,
            self.empty_add_button,
            self.mode_combo,
            self.compute_combo,
            self.gpu_combo,
            self.ram_spin,
            self.vram_spin,
            self.context_spin,
            self.max_tokens_spin,
            self.port_spin,
            self.python_edit,
            self.python_browse_button,
        ):
            widget.setEnabled(enabled)
        has_model = enabled and self.model_list.count() > 0
        self.rename_model_button.setEnabled(has_model)
        self.remove_model_button.setEnabled(has_model)
        self._update_gpu_control_enabled()

    def _invalidate_pending_launch(self) -> None:
        if not self._launch_in_progress or self._supervisor is not None:
            return
        self._launch_generation += 1
        self._launch_request = None
        self._pending_start = False
        self._launch_in_progress = False
        self._set_launch_controls_enabled(True)

    def _finish_launch_operation(self) -> None:
        self._pending_start = False
        self._launch_request = None
        self._launch_in_progress = False
        self._set_launch_controls_enabled(True)

    def _options_changed(self, *_args: Any) -> None:
        if self._updating_options:
            return
        self._invalidate_pending_launch()
        self._update_gpu_control_enabled()
        entry = self._selected_entry()
        if entry is None:
            return
        entry["options"] = dataclasses.asdict(self._options())
        self._save()
        self._preflight = None
        self._preflight_key = None
        self.start_button.setEnabled(False)
        self.readiness_label.setText("Settings changed. Checking again…")
        self._preflight_timer.start()

    def _begin_preflight(self, for_start: bool = False) -> None:
        entry = self._selected_entry()
        if entry is None or self._installation is None:
            return
        options = self._options()
        path = str(entry["path"])
        installation = self._installation
        generation = self._installation_generation
        self._pending_start = self._pending_start or for_start
        self.start_button.setEnabled(False)
        self.retry_button.hide()
        self.readiness_label.setText("Checking model and hardware…")
        self._run(
            "preflight",
            inspect_model,
            installation,
            path,
            options,
            on_result=lambda result: self._preflight_ready(
                path, options, installation, generation, result
            ),
            on_error=self._preflight_failed,
        )

    def _preflight_ready(
        self,
        path: str,
        options: LaunchOptions,
        installation: Any,
        generation: int,
        result: Preflight,
    ) -> None:
        current = self.model_list.currentItem()
        if (
            current is None
            or current.data(Qt.ItemDataRole.UserRole) != path
            or self._options() != options
            or self._installation != installation
            or self._installation_generation != generation
        ):
            return
        self._preflight = result
        self._preflight_key = (path, options)
        self._updating_options = True
        try:
            self._set_gpu_choices(result.gpus, options.gpu_ids)
            self.context_spin.setMaximum(max(1, result.model.max_context))
            self.context_spin.setToolTip(f"Automatic uses {result.model.default_context:,} tokens")
            max_output = result.plan.get("max_output", result.model.max_context)
            if not isinstance(max_output, int) or isinstance(max_output, bool) or max_output < 1:
                max_output = result.model.max_context
            self.max_tokens_spin.setMaximum(max_output)
            self.max_tokens_spin.setToolTip(f"Automatic uses {result.model.default_output:,} tokens")
        finally:
            self._updating_options = False
        cuda_index = self.compute_combo.findData("cuda")
        cuda_item = self.compute_combo.model().item(cuda_index)
        # CPU diagnostics do not verify CUDA readiness. Allow choosing it when
        # the engine and devices support it; that choice runs fresh diagnostics.
        cuda_item.setEnabled(result.cuda_available or
                             (options.compute == "cpu" and result.plan.get("cuda_capable") is True))
        detected = "\n".join(f"Detected GPU {gpu.index}: {gpu.name}" for gpu in result.gpus)
        self.compute_reason.setText(f"{detected}\n{result.cuda_reason}" if detected else result.cuda_reason)
        explicit_cuda_unavailable = not result.cuda_available and options.compute == "cuda"
        if explicit_cuda_unavailable:
            self._pending_start = False
            self._finish_launch_operation()
        messages = [check.message for check in result.checks if check.level in {"fail", "warn"}]
        if explicit_cuda_unavailable and result.cuda_reason not in messages:
            messages.insert(0, result.cuda_reason)
        can_start = result.can_start and not explicit_cuda_unavailable
        if can_start:
            self.readiness_label.setText(messages[0] if messages else "Ready to start.")
            self.retry_button.hide()
            self.start_button.setEnabled(
                not self._launch_in_progress
                and (self._supervisor is None or self._supervisor.state in {"stopped", "failed"})
            )
        else:
            self.readiness_label.setText("\n".join(messages) or "This model is not ready yet.")
            self.retry_button.show()
            self.start_button.setEnabled(False)
        if self._pending_start:
            self._pending_start = False
            if can_start:
                self._build_and_start(result, options)
            else:
                self._finish_launch_operation()

    def _preflight_failed(self, message: str) -> None:
        self._pending_start = False
        self._preflight = None
        self._finish_launch_operation()
        self.readiness_label.setText(message)
        self.retry_button.show()
        self.start_button.setEnabled(False)

    def _retry(self) -> None:
        if self._installation is None:
            self._discover_installation()
        else:
            self._begin_preflight()

    def _start(self) -> None:
        if self._installation is None or self._python_edit_pending or self._launch_in_progress or (
            self._supervisor is not None and self._supervisor.state not in {"stopped", "failed"}
        ):
            return
        self._launch_in_progress = True
        self._stop_requested = False
        self._set_launch_controls_enabled(False)
        self._pending_start = True
        self._begin_preflight(for_start=True)

    def _build_and_start(self, result: Preflight, options: LaunchOptions) -> None:
        entry = self._selected_entry()
        if entry is None or self._installation is None:
            self._finish_launch_operation()
            return
        self._launch_generation += 1
        request = (self._launch_generation, str(entry["path"]), self._installation, options)
        self._launch_request = request
        self.status_label.setText("Preparing")
        self.start_button.setEnabled(False)
        self._run(
            "launch",
            build_launch,
            self._installation,
            result,
            options,
            on_result=lambda spec: self._launch_ready(request, spec),
            on_error=self._launch_failed,
        )

    def _launch_ready(self, request: tuple[int, str, Any, LaunchOptions], spec: Any) -> None:
        current = self.model_list.currentItem()
        request_matches = (
            self._launch_in_progress
            and self._launch_request == request
            and current is not None
            and current.data(Qt.ItemDataRole.UserRole) == request[1]
            and self._installation == request[2]
            and self._options() == request[3]
        )
        supervisor_active = self._supervisor is not None and self._supervisor.state not in {"stopped", "failed"}
        if not request_matches or supervisor_active:
            return
        self.command_preview_edit.setText(command_preview(spec))
        self._supervisor = Supervisor(self._event_bridge.publish)
        self._requested_compute = request[3].compute
        self._cuda_fallback_observed = False
        self._chat_opened = False
        self._load_started = time.monotonic()
        self._runtime_state = "starting"
        self._elapsed_timer.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Loading")
        self._supervisor_start_pending = True
        self._run(
            "supervisor-start",
            self._start_supervisor,
            spec,
            on_result=self._supervisor_start_finished,
            on_error=self._launch_failed,
        )

    def _start_supervisor(self, spec: Any) -> None:
        supervisor = self._supervisor
        if supervisor is None:
            return
        supervisor.start(spec)
        if self._stop_requested:
            supervisor.stop()
            supervisor.wait(10.0)

    def _supervisor_start_finished(self, _result: Any) -> None:
        self._supervisor_start_pending = False

    def _launch_failed(self, message: str) -> None:
        self._supervisor_start_pending = False
        self._finish_launch_operation()
        self.status_label.setText("Could not start")
        self.readiness_label.setText(message)
        self.retry_button.show()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(self._preflight is not None and self._preflight.can_start)

    def _stop(self) -> None:
        if self._supervisor is None:
            return
        self._stop_requested = True
        self.status_label.setText("Stopping")
        self.stop_button.setEnabled(False)
        if self._supervisor_start_pending and self._supervisor.state == "stopped":
            return
        self._run(
            "supervisor-stop",
            self._stop_supervisor,
            on_result=lambda _result: None,
            on_error=self._launch_failed,
        )

    def _stop_supervisor(self) -> None:
        supervisor = self._supervisor
        if supervisor is None:
            return
        supervisor.stop()
        supervisor.wait(10.0)

    @Slot(object)
    def _on_process_event(self, event: ProcessEvent) -> None:
        if not self._alive:
            return
        if event.kind == "log":
            self._append_log(event.text)
            return
        if event.kind == "error":
            self._append_log(event.text)
            self.status_label.setText("Failed")
            self.readiness_label.setText(event.text)
            self.retry_button.show()
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(self._preflight is not None and self._preflight.can_start)
            return
        if event.kind != "state":
            return
        state = event.text.strip().lower()
        if self._runtime_state == "running" and state in {"starting", "loading"}:
            return
        self._runtime_state = state
        if state in {"starting", "loading"}:
            if self._stop_requested:
                self.status_label.setText("Stopping")
                self.stop_button.setEnabled(False)
            else:
                self.status_label.setText("Loading")
                self.stop_button.setEnabled(True)
        elif state == "running":
            if self._stop_requested:
                self.status_label.setText("Stopping")
                self.stop_button.setEnabled(False)
                return
            self._elapsed_timer.stop()
            elapsed = max(0, int(time.monotonic() - self._load_started))
            self.elapsed_label.setText(f"Ready after {elapsed // 60}:{elapsed % 60:02d}")
            self.stop_button.setEnabled(True)
            spec = self._supervisor.spec if self._supervisor else None
            if spec is not None:
                self._show_runtime_readiness(spec)
                if spec.mode == "serve":
                    self.api_address.setText(f"http://127.0.0.1:{spec.port}  •  model: {spec.model_id}")
                    self.api_address.show()
                elif not self._chat_opened:
                    self._chat_opened = True
                    webbrowser.open(f"http://127.0.0.1:{spec.port}")
        elif state == "stopping":
            self.status_label.setText("Stopping")
            self.stop_button.setEnabled(False)
        elif state in {"stopped", "failed"}:
            self._elapsed_timer.stop()
            self._supervisor_start_pending = False
            self._finish_launch_operation()
            self.status_label.setText("Stopped" if state == "stopped" else "Failed")
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(self._preflight is not None and self._preflight.can_start)
            if self._close_after_stop:
                self._allow_close = True
                self.close()
            self._stop_requested = False

    def _show_runtime_readiness(self, spec: Any) -> None:
        if self._cuda_fallback_observed:
            if self._requested_compute == "cuda":
                self.status_label.setText("Backend mismatch")
                self.readiness_label.setText(
                    "CUDA was explicitly requested, but the server reported a CPU fallback. "
                    "Stop the model and review Details."
                )
            else:
                self.status_label.setText("Ready")
                self.readiness_label.setText(
                    "Server ready • Automatic compute fell back to CPU. Review Details for the reported reason."
                )
            return
        self.status_label.setText("Ready")
        if spec.backend == "cuda":
            backend_summary = "NVIDIA CUDA requested. Check Details for runtime confirmation."
        else:
            backend_summary = "CPU configured."
        self.readiness_label.setText(f"Server ready • {backend_summary}")

    def _append_log(self, text: str) -> None:
        for line in str(text).splitlines() or [str(text)]:
            self._log_lines.append(line[:8192])
            self.log_view.appendPlainText(line[:8192])
            normalized = line.lower()
            cuda_fallback = (
                "cuda unavailable" in normalized
                or "cuda disabled" in normalized
                or ("fall" in normalized and "back" in normalized and "cpu" in normalized)
                or ("device unavailable" in normalized and "stay on cpu" in normalized)
            )
            if (
                cuda_fallback
                and self._supervisor is not None
                and self._supervisor.spec is not None
                and self._supervisor.spec.backend == "cuda"
            ):
                self._cuda_fallback_observed = True
                if self._runtime_state == "running":
                    self._show_runtime_readiness(self._supervisor.spec)
                else:
                    self.readiness_label.setText(
                        "The server reported that CUDA may have fallen back to CPU. "
                        "Runtime confirmation is still pending."
                    )

    def _update_elapsed(self) -> None:
        if self._load_started:
            elapsed = int(time.monotonic() - self._load_started)
            self.elapsed_label.setText(f"Loading for {elapsed // 60}:{elapsed % 60:02d}")

    def _copy_command(self) -> None:
        QGuiApplication.clipboard().setText(self.command_preview_edit.text())

    def _export_logs(self) -> None:
        filename, _selected_filter = QFileDialog.getSaveFileName(
            self, "Export launcher logs", "colibri-launcher.log", "Log files (*.log);;Text files (*.txt)"
        )
        if not filename:
            return
        Path(filename).write_text("\n".join(self._log_lines), encoding="utf-8")

    def _browse_python(self) -> None:
        filename, _selected_filter = QFileDialog.getOpenFileName(self, "Choose Python")
        if filename:
            self.python_edit.setText(filename)
            self._discover_installation()

    def _toggle_advanced(self, shown: bool) -> None:
        self._toggle_panel(self.advanced_toggle, self.advanced_panel, shown)

    def _toggle_details(self, shown: bool) -> None:
        self._toggle_panel(self.details_toggle, self.details_panel, shown)

    @staticmethod
    def _toggle_panel(button: QToolButton, panel: QWidget, shown: bool) -> None:
        panel.setVisible(shown)
        button.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)

    def _update_empty_state(self) -> None:
        empty = self.model_list.count() == 0
        self.empty_state_card.setVisible(empty)
        self.controls_frame.setVisible(not empty)
        self.rename_model_button.setEnabled(not empty)
        self.remove_model_button.setEnabled(not empty)

    @Slot(int)
    def _change_theme(self, _index: int) -> None:
        theme = self.theme_combo.currentData()
        previous = self._settings.get("theme", "light")
        self._settings["theme"] = theme
        try:
            self._save(raise_on_error=True)
        except LauncherError as error:
            self._settings["theme"] = previous
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(self.theme_combo.findData(previous))
            self.theme_combo.blockSignals(False)
            QMessageBox.warning(self, "Theme could not be saved", str(error))
            return
        apply_theme(QApplication.instance(), theme)
        self.setStyleSheet(stylesheet(theme))

    def _save(self, *, raise_on_error: bool = False) -> None:
        try:
            save_settings(self._settings, self.settings_path)
        except LauncherError as error:
            if raise_on_error:
                raise
            self.settings_warning_label.setText(
                f"{error}\nChanges remain available for this session only until saving succeeds."
            )
            self.settings_warning_label.show()
        else:
            self.settings_warning_label.hide()

    def _invalidate_worker_category(self, category: str) -> None:
        self._request_number += 1
        self._latest[category] = f"{category}:invalid:{self._request_number}"

    def _run(
        self,
        category: str,
        function: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._request_number += 1
        token = f"{category}:{self._request_number}"
        self._latest[category] = token
        self._callbacks[token] = (on_result, on_error)
        worker = FunctionWorker(token, function, *args)
        worker.signals.result.connect(self._worker_result)
        worker.signals.failed.connect(self._worker_failed)
        self._workers.start(worker)

    @Slot(str, object)
    def _worker_result(self, token: str, value: Any) -> None:
        callback_pair = self._callbacks.pop(token, None)
        category = token.rsplit(":", 1)[0]
        if not self._alive or callback_pair is None or self._latest.get(category) != token:
            return
        callback_pair[0](value)

    @Slot(str, str)
    def _worker_failed(self, token: str, message: str) -> None:
        callback_pair = self._callbacks.pop(token, None)
        category = token.rsplit(":", 1)[0]
        if not self._alive or callback_pair is None or self._latest.get(category) != token:
            return
        callback_pair[1](message)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            self._alive = False
            self._callbacks.clear()
            event.accept()
            return
        active = self._supervisor is not None and (
            self.stop_button.isEnabled() or self._supervisor.state not in {"stopped", "failed"}
        )
        if active:
            answer = QMessageBox.question(
                self,
                "Stop model and quit?",
                "Colibri is still loading or running. Stop it before closing the launcher?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_after_stop = True
            self._stop()
            event.ignore()
            return
        self._alive = False
        self._callbacks.clear()
        event.accept()
