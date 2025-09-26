# Brans-Dicke_vs_-CDM
Análisis cosmológico completo que compara la teoría de Brans-Dicke con el modelo estándar ΛCDM, utilizando tanto datos sintéticos como observaciones reales del survey Pantheon de supernovas Type Ia.
🌌 Brans-Dicke vs ΛCDM Cosmological Analysis
https://img.shields.io/badge/python-3.8%252B-blue
https://img.shields.io/badge/License-CC_BY_4.0-lightgrey
https://img.shields.io/badge/arXiv-2509.XXXXX-b31b1b

📖 Descripción
Análisis cosmológico completo que compara la teoría de gravitación de Brans-Dicke con el modelo estándar ΛCDM, utilizando datos sintéticos y observaciones reales del survey Pantheon de supernovas Type Ia.

🚀 Características Principales
🔬 Análisis Numérico: Solución de ecuaciones de campo de Brans-Dicke

📊 Comparación Estadística: Test χ² entre modelos cosmológicos

🌐 Datos Multi-fuente: Sintéticos + Pantheon supernova survey

📈 Visualización Avanzada: Gráficos profesionales para publicación

🔄 Código Reproducible: Implementación completa en Python

📁 Estructura del Proyecto
text
Brans-Dicke_vs_-CDM/
│
├── 📁 code/                          # Código fuente Python
│   ├── 📄 cosmological_analysis.py   # Análisis principal
│   ├── 📄 brans_dicke_solver.py      # Solucionador de ecuaciones BD
│   ├── 📄 data_loader.py             # Carga de datos Pantheon
│   └── 📄 visualization.py           # Funciones de gráficos
│
├── 📁 results/                       # Resultados generados
│   ├── 📁 figures/                   # Gráficos de análisis
│   ├── 📁 tables/                    # Tablas de parámetros
│   ├── 📁 data/                      # Datos procesados
│   └── 📁 statistical/               # Análisis estadístico
│
├── 📁 manuscript/                    # Documentación científica
│   ├── 📄 manuscript.pdf             # Manuscrito final
│   ├── 📄 manuscript.tex             # Código LaTeX
│   └── 📄 references.bib             # Bibliografía
│
├── 📁 notebooks/                     # Jupyter notebooks
│   ├── 📄 main_analysis.ipynb        # Análisis interactivo
│   └── 📄 tutorial.ipynb             # Tutorial introductorio
│
└── 📄 README.md                      # Este archivo
🛠️ Instalación Rápida
1. Clonar el repositorio
bash
git clone https://github.com/miguelpercu/Brans-Dicke_vs_-CDM.git
cd Brans-Dicke_vs_-CDM
2. Instalar dependencias
bash
pip install -r requirements.txt
3. Ejecutar análisis completo
bash
python code/cosmological_analysis.py
📊 Resultados Científicos
🔍 Hallazgos Principales
Escenario	Mejor Ajuste	χ²	Δχ² vs ΛCDM	Conclusión
Datos Sintéticos	ω = 10	24.53	+0.06	Equivalente estadístico
Datos Reales (Pantheon)	ω = 10	1558.79	+521.61	ΛCDM preferido
📈 Gráficos Generados
comprehensive_analysis.pdf - Análisis completo con datos sintéticos

real_data_analysis.pdf - Validación con datos Pantheon

chi2_comparison.pdf - Comparación estadística detallada

🎯 Uso Rápido
Desde Python
python
from code.cosmological_analysis import main_analysis

# Ejecutar análisis completo
results, lcdm_results, best_fit = main_analysis()

# Acceder a resultados
print(f"ΛCDM χ²: {lcdm_results[0]:.2f}")
print(f"Mejor BD (ω={best_fit[0]}): χ² = {best_fit[1]:.2f}")
Desde Jupyter Notebook
python
%run code/cosmological_analysis.py
# o abrir notebooks/main_analysis.ipynb
⚙️ Parámetros Configurables
Cosmológicos
python
cosmo_params = {
    'H0': 70.0,           # km/s/Mpc
    'Omega_m0': 0.3,      # Densidad materia
    'Omega_DE0': 0.7,     # Densidad energía oscura
    'w0': -1.0            # Ecuación de estado
}
Brans-Dicke (valores probados)
python
omega_values = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
📚 Marco Teórico
Ecuaciones Implementadas
Brans-Dicke (Brans & Dicke, 1961):

math
H^2 = \frac{8\pi\rho}{3\phi} - \frac{\dot{\phi}}{\phi}H + \frac{\omega}{6}\left(\frac{\dot{\phi}}{\phi}\right)^2
ΛCDM (caso límite ω→∞):

math
H^2 = H_0^2[\Omega_m(1+z)^3 + \Omega_\Lambda]
🔬 Métodos Estadísticos
Test χ² con marginalización de normalización

Comparación de modelos via Δχ²

Validación cruzada sintético/real

Análisis de residuos normalizados

🎓 Aplicaciones Educativas
Ideal para:

Cursos de cosmología avanzada

Investigación en gravitación modificada

Aprendizaje de análisis numérico en astrofísica

Metodología de comparación de modelos científicos

🤝 Contribuir
¡Contribuciones bienvenidas! Áreas de interés:

Mejoras Numéricas

Solvers más eficientes

Manejo de singularidades

Extensiones Teóricas

Teorías f(R)

Gravitación masiva

Modelos G-c covariantes

Nuevos Análisis

Datos CMB + BAO

Análisis bayesiano (MCMC)

Constraints de lensing

📄 Citación
Si usas este código en tu investigación, por favor cita:

bibtex
@software{percudani2025bransdicke,
  author = {Percudani, Miguel Angel},
  title = {Brans-Dicke vs ΛCDM Cosmological Analysis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/miguelpercu/Brans-Dicke_vs_-CDM}}
}
👨‍🔬 Autor
Miguel Angel Percudani
Investigador Independiente
📧 miguel_percudani@yahoo.com.ar
📍 Puan, Buenos Aires, Argentina

🔗 Enlaces Relacionados
📄 Manuscrito Completo - Paper científico detallado

🐍 Tutorial Interactivo - Introducción paso a paso

📊 Resultados Completos - Todos los análisis y gráficos

📜 Licencia
Este proyecto está bajo la Licencia Creative Commons Attribution 4.0 International - ve el archivo LICENSE para detalles.

⭐ ¿Encontraste útil este proyecto? ¡Dale una estrella al repositorio!

Última actualización: Septiembre 2025
¿Preguntas? Abre un issue o contáctame directamente.
