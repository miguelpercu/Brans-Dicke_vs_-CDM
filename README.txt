📊 Cosmological Analysis: Brans-Dicke vs ΛCDM
🌌 Descripción del Proyecto
Análisis cosmológico completo que compara la teoría de Brans-Dicke con el modelo estándar ΛCDM, utilizando tanto datos sintéticos como observaciones reales del survey Pantheon de supernovas Type Ia.

🚀 Características Principales
Análisis numérico de las ecuaciones de Brans-Dicke

Comparación estadística con el modelo ΛCDM

Datos sintéticos y reales (Pantheon supernova survey)

Análisis χ² con marginalización de parámetros

Visualización comprehensiva de resultados

Código reproducible y documentado

📁 Estructura del Proyecto
text
cosmology-brans-dicke/
│
├── 📊 results/                          # Resultados del análisis
│   ├── figures/                         # Gráficos generados
│   │   ├── comprehensive_analysis.pdf   # Análisis principal
│   │   ├── real_data_analysis.pdf       # Datos reales Pantheon
│   │   └── chi2_comparison.pdf          # Comparación estadística
│   ├── tables/                          # Tablas de parámetros
│   ├── data_observations/               # Datos observacionales
│   ├── data_curves/                     # Curvas teóricas
│   └── statistical_analysis/            # Resultados estadísticos
│
├── 📄 Manuscrito/                       # Documentación científica
│   ├── manuscript.pdf                   # Manuscrito final
│   ├── manuscript.tex                   # Código LaTeX
│   └── references.bib                   # Bibliografía
│
└── 🐍 Código/                           # Implementación Python
    ├── main_analysis.py                 # Análisis principal
    ├── brans_dicke_solver.py            # Solucionador de ecuaciones
    └── visualization.py                 # Funciones de visualización
🔬 Métodos Científicos
Teorías Implementadas
ΛCDM: Modelo cosmológico estándar

Brans-Dicke: Teoría de gravitación modificada con campo escalar

Análisis Estadístico
Método χ² con marginalización de normalización

Comparación de modelos mediante Δχ²

Validación cruzada con datos sintéticos y reales

📈 Resultados Clave
Con Datos Sintéticos
Equivalencia estadística entre BD y ΛCDM (Δχ² = +0.06)

Mejor ajuste: ω = 10

χ²_reducido: 0.876 (excelente ajuste)

Con Datos Reales (Pantheon)
Fuerte preferencia por ΛCDM (Δχ² = +521.61)

BD muestra discrepancias significativas

χ²_reducido: 1.489 vs 0.991 (ΛCDM)

🛠️ Requisitos del Sistema
Python 3.8+
bash
pip install numpy matplotlib pandas scipy jupyter
Paquetes Principales
numpy >= 1.21.0

matplotlib >= 3.5.0

pandas >= 1.4.0

scipy >= 1.8.0

🚀 Ejecución Rápida
1. Análisis Completo
python
from main_analysis import main_analysis
results, lcdm_results, best_fit = main_analysis()
2. Solo Visualización
python
from visualization import create_comprehensive_plots
create_comprehensive_plots(z_data, mu_obs, mu_err, results, lcdm_results, best_fit)
3. Desde Jupyter Notebook
python
%run main_analysis.py
📊 Parámetros Cosmológicos
Parámetro	Valor	Descripción
H₀	70.0 km/s/Mpc	Constante de Hubble
Ωₘ₀	0.3	Densidad de materia
Ω_DE₀	0.7	Densidad de energía oscura
w₀	-1.0	Ecuación de estado DE
🔍 Parámetros Brans-Dicke Probados
ω = [10, 50, 100, 500, 1000, 2000, 5000, 10000]

📚 Referencias Científicas
Brans & Dicke (1961): Teoría original de Brans-Dicke

Planck Collaboration (2020): Parámetros cosmológicos

Scolnic et al. (2018): Survey Pantheon de supernovas

Will (2006): Constraints del sistema solar

🎯 Aplicaciones y Extensiones
Próximos Pasos de Investigación
Análisis con datos Pantheon+

Incorporación de constraints de CMB

Extensión a teorías G-c covariantes

Análisis bayesiano con MCMC

Aplicaciones Educativas
Enseñanza de cosmología numérica

Comparación de teorías gravitacionales

Análisis estadístico en cosmología

🤝 Contribuciones
Este proyecto es de código abierto. Contribuciones welcome en:

Mejoras numéricas del solver

Nuevos análisis estadísticos

Extensiones teóricas

Documentación

📄 Licencia
Creative Commons Attribution 4.0 International

👨‍🔬 Autor
Miguel Angel Percudani
Investigador Independiente
Puan, Buenos Aires, Argentina
Email: miguel_percudani@yahoo.com.ar

🔗 Enlaces Relacionados
Repositorio GitHub: https://github.com/miguelpercu/Brans-Dicke_vs_-CDM

Documentación Completa