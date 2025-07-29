# Proyecto IR 2025A

Este proyecto es una aplicación desarrollada en Python para la materia de Información y Recuperación (IR) del periodo 2025A en la EPN.

## Estructura del proyecto
- `src/`: Contiene el código fuente principal del proyecto.
  - `actuator/`: Módulo relacionado con la gestión de actuadores (health endpoints).
  - `ir/`: Módulo principal para la lógica de recuperación de información.
  - `main.py`: Punto de entrada de la aplicación.
  - `api.py`: Define la API del proyecto.
- `pyproject.toml` y `poetry.lock`: Archivos de configuración y dependencias del proyecto.

## Requisitos
- Python 3.12 o 3.13
- [Poetry](https://python-poetry.org/) para la gestión de dependencias


## Instalación
1. Clona este repositorio.
2. Localizate en la carpeta del proyecto
3. Instala las dependencias del proyecto usando [Poetry](https://python-poetry.org/docs/#installation):
   ```bash
   poetry install
   ```

## Uso
Para ejecutar la aplicación principal:
```bash
poetry run poe api
```

## URLs de la API
- `http://127.0.0.1:8000/docs/`: Docs.
- `http://127.0.0.1:8000/actuator/health/`: Health Endpoint.

## Licencia
Este proyecto es solo para fines educativos.
