<center>
<img src="./images/logo-inpe.png" width="100" />
</center>

# INPE CAPPY239 - Análise Espectral de Processos Estocásticos
Ferramenta desenvolvida para a disciplina CAP 238 Matemática Computacional I - Análise Espectral de Processos  Estocásticos, ministrada pelo Dr. Reinaldo Roberto Rosa no programa de Pós Graduação em Computação Aplicada no Instituto Nacional de Pesquisas Espaciais.
O módulo possui as seguintes funções:

## Instalação
    pip install cappy239

> Disponível para Python 3.0 ou superior.

## Powernoise
A função `powernoise`  gera sinais estocásticos, ruídos 1/f<sup>β</sup>. É uma adaptação do código implementado em Matlab pelo Dr. Reinaldo R. Rosa e Dale B. Dalrymple.

Alguns exemplos com 2<sup>12</sup> valores:
 - **White noise**:
```python
from cappy239 import powernoise
white_noise = powernoise(beta=0, N=4096)
```
 - **Pink noise**:
```python
from cappy239 import powernoise
pink_noise = powernoise(beta=1, N=4096)
```
 - **Red noise**:
```python
from cappy239 import powernoise
red_noise = powernoise(beta=2, N=4096)
```

> Por padrão s série temporal está normalizada, caso não queria isso passe o argumento `varargin='randpower'`.

## Mapeamento Quadrático (Logístico)
A função `logistic_map` gera uma série caótica . 
Exemplo com 100 valores:
```python
from cappy239 import logistic_map
chaotic_serie = logistic_map(rho=3.85, a0=0.001, n=100)
```
> Referência: [https://geoffboeing.com/2015/03/chaos-theory-logistic-map/](https://geoffboeing.com/2015/03/chaos-theory-logistic-map/)

## Pmodel
A função `pmodel` é utilizada para gerar séries temporais estacionárias. O código é uma adaptação da implementação em Matlab disponível em [http://www2.meteo.uni-bonn.de/staff/venema/themes/surrogates/pmodel/pmodel.m](http://www2.meteo.uni-bonn.de/staff/venema/themes/surrogates/pmodel/pmodel.m).
Alguns exemplos com 2<sup>12</sup> valores:

 - **S8: p=0.52, β=-1.66**
```python
from cappy239 import pmodel
kolmogorov = pmodel(noValues=4096, p=0.52, slope=-1.66)
```

   ______       _       _______  _______  ____  ____  _____   ______    ______   
 .' ___  |     / \     |_   __ \|_   __ \|_  _||_  _|/ ___ `./ ____ `..' ____ '. 
/ .'   \_|    / _ \      | |__) | | |__) | \ \  / / |_/___) |`'  __) || (____) | 
| |          / ___ \     |  ___/  |  ___/   \ \/ /   .'____.'_  |__ '.'_.____. | 
\ `.___.'\ _/ /   \ \_  _| |_    _| |_      _|  |_  / /_____| \____) || \____| | 
 `.____ .'|____| |____||_____|  |_____|    |______| |_______|\______.' \______,' 
                                                           By Adriano P. Almeida                                                                                 
