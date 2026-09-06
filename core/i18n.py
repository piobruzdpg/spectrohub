"""Menedżer lokalizacji i tłumaczeń."""

from PySide6.QtCore import QObject, Signal, QSettings
from locales import pl, en


class I18nManager(QObject):
    language_changed = Signal()

    def __init__(self):
        super().__init__()
        self._settings = QSettings("SpectroHub", "Preferences")
        # Domyślnie bierzemy z ustawień lub ustawiamy 'pl'
        self._current_lang = self._settings.value("language", "pl")

        self._dictionaries = {
            "pl": pl.TRANSLATIONS,
            "en": en.TRANSLATIONS
        }

    @property
    def current_language(self) -> str:
        return self._current_lang

    def set_language(self, lang_code: str):
        if lang_code in self._dictionaries and lang_code != self._current_lang:
            self._current_lang = lang_code
            self._settings.setValue("language", lang_code)
            self.language_changed.emit()

    def tr(self, key_path: str, default: str = "") -> str:
        """
        Pobiera tłumaczenie po zagnieżdżonym kluczu, np. 'deconv.table.center'.
        W razie braku klucza zwraca domyślną wartość lub sam klucz.
        """
        keys = key_path.split(".")
        current_dict = self._dictionaries.get(self._current_lang, {})

        val = current_dict
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default or key_path

        return str(val) if val is not None else (default or key_path)


# Globalna instancja menedżera oraz alias 'tr' do bezpośredniego importu
i18n = I18nManager()
tr = i18n.tr