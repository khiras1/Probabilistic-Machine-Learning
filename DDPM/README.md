# Assignment l09

Repozytorium zawiera notebooki zawierające zadanie na ten tydzień. Wykonaj polecenia w każdym notebooku (rozszerzenie .ipynb). Następnie zacommituj pracę i wypushuj na GitHuba. Prowadzący sprawdzi wykonane zadanie po jego terminie.


Wyniki zostaną opublikowane w plikach feedbacku `*.html`. Konieczne będzie pobranie zmian i otwarcie pliku w przeglądarce (GitHub nie renderuje plików HTML).

----

## Konfiguracja środowiska

W laboratorium wykorzystywane są zeszyty [Jupyter Notebook](https://jupyter.org/). Są to interaktywne dokumenty mogące zawierać kod oraz wyniki jego działania, wraz z dodatkową treścią, dodawaną przy użyciu języka [Markdown](https://www.markdownguide.org/). W zeszytach Jupyter Notebook można wykorzystywać wiele języków; na laboratorium wykorzystywany będzie język Python.

Przygotowanie środowiska polega na zainstalowaniu wymaganych paczek z dostarczonego pliku `requirements.txt`. Można to zrobić na kilka sposobów:

1. Używając pip w wirtualnym środowisku:
   - Utwórz wirtualne środowisko: `python -m venv myenv`
   - Aktywuj środowisko: `source myenv/bin/activate` (Linux/macOS) lub `myenv\Scripts\activate` (Windows)
   - Zainstaluj paczki: `pip install -r requirements.txt`

2. Używając Conda:
   - Utwórz nowe środowisko: `conda create --name myenv`
   - Aktywuj środowisko: `conda activate myenv`
   - Zainstaluj paczki: `pip install -r requirements.txt`

3. Używając narzędzia takiego jak `pipenv` lub `poetry` do zarządzania środowiskami i zależnościami.

4. Używając `uv`:
   - Zainstaluj `uv`: https://github.com/astral-sh/uv 
   - Utwórz nowe środowisko: `uv new myenv`
   - Aktywuj środowisko: `uv activate myenv`
   - Zainstaluj paczki: `uv add -r requirements.txt`
