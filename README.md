# Modelagem Covid-19 - IM/UFRJ - Período 2020/1

[![Text License: CC-BY-NC-ND license](https://img.shields.io/badge/Text%20License-CC--BY--NC--ND-yellow.svg)](https://opensource.org/licenses/MIT) [![Code License: GNU-GPLv3](https://img.shields.io/badge/Code%20License-GNU--GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl.html) ![GitHub repo size](https://img.shields.io/github/repo-size/rmsrosa/nbbinder)

Notas sobre modelagem matemática da epidemia de Covid-19.

[Ricardo M. S. Rosa](http://www.im.ufrj.br/rrosa/).

## *Link* de acesso direto aos cadernos no github

*Link* para acessar a [página inicial]((contents/notebooks/00.00-Pagina_Inicial.ipynb)) das notas no [Github](https://github.com):

[![Github](https://img.shields.io/badge/view%20contents%20on-github-orange)](contents/notebooks/00.00-Pagina_Inicial.ipynb)

## Outros *links* de acesso direto aos cadernos

ESSES LINKS SÓ FUNCIONAM QUANDO O REPOSITÓRIO É PÚBLICO!

*Links* para acessar a página inicial das notas via [Binder](https://beta.mybinder.org/) e [Google Colab](http://colab.research.google.com):

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rmsrosa/modcovid19/master?filepath=contents%2Fnotebooks%2F00.00-Pagina_Inicial.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rmsrosa/modcovid19/blob/master/contents/notebooks/00.00-Pagina_Inicial.ipynb)

## Observações

- As **notas de aula** estão dispostas na forma de uma coleção de [cadernos Jupyter](https://jupyter.org/) e estão disponíveis no subdiretório [contentes/notebooks](contents/notebooks). As notas serão escritas e disponibilizadas ao longo do curso/projeto.

- Há uma [Página Inicial](contents/notebooks/00.00-Pagina_Inicial.ipynb) exibindo a coleção de cadernos de forma estruturada, com links para cada caderno.

- Os **anexos (imagens, dados)** estão em no subdiretório [contents/input](contents/input).

- Alguns **artigos** relevantes em [contents/bibliography](contents/bibliography)

- Uma **biblioteca** python com códigos para a simulação de epidemia usando variados modelos está sendo elaborada no subdiretório [contents/episiming](contents/episiming).

- Um servidor [Jupyter Hub](https://jupyter.org/hub) próprio para o curso está disponível em [Jupyter Hub ModCovid19](https://rmsrosa.tk/jupyter/),  onde cada aluno terá uma conta e todo o ambiente computacional necessário para a execução dos notebooks.

- O servidor [Jupyter Hub ModCovid19](https://rmsrosa.tk/jupyter/) foi instalado em uma instância da [Amazon EC2 (Amazon Elastic Compute Cloud)](https://aws.amazon.com/pt/ec2/), do tipo [t2.micro](https://aws.amazon.com/pt/ec2/instance-types/), com apenas 1 vCPU, 1Gb RAM, 8Gb SSD. Não é muito, mas serve para uma primeira experiência e serve para a execução dos trabalhos a serem entregues via [nbgrader](https://nbgrader.readthedocs.io/). Mas, para uso geral, recomendamos uma máquina local ou uma das outras nuvens de computação ([Binder](https://beta.mybinder.org/), [Google Colab](http://colab.research.google.com), [Kaggle](https://www.kaggle.com/), por exemplo).

- Cada aluno deverá enviar, para o email do professor, o seu **nome**, **sobrenome**, **email** e o **nome de usuário** que deseja ter no [Jupyter Hub ModCovid19](https://www.modcovid19.ricardomsrosa.org/jupyter).

- Ao longo do período, é esperado que os alunos modifiquem os cadernos existentes e criem os seus próprios cadernos para resolver os exercícios, os testes e escrever os trabalhos/mini-projetos e o projeto final.

- Todo o conteúdo do repositório pode ser baixado para uma máquina local através do botão `Clone or Download` da página inicial [rmsrosa/modcovid19](https://github.com/rmsrosa/modcovid19) e escolhendo a opção `Download ZIP`.

- Cada caderno também pode ser baixado individualmente para uma máquina local clicando-se no ícone `Raw`. O conteúdo que aparecer no navegador é um arquivo fonte de cadernos [jupyter](https://jupyter.org/), com a extensão `".ipynb"`.

- As alterações nos cadernos deste repositório e a criação de novos cadernos podem ser feitas localmente, em máquinas com o Python versão 3.8 e os devidos pacotes devidamente instalados.

- A lista dos pacotes python necessários para a execução do conjunto de cadernos aparece no arquivo [requirements.txt](requirements.txt). Esse arquivo não é apenas uma referência, ele é necessário caso deseja-se acessar o notebook no [Binder](https://beta.mybinder.org/), mas para isso o repositório deve estar com acesso público.

## Licença

Os **textos** neste repositório estão disponíveis sob a licença [CC-BY-NC-ND license](LICENSE-TEXT). Mais informações sobre essa licença em [Creative Commons Attribution-NonCommercial-NoDerivs 3.0](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode).

Os **códigos** neste repositório, nos blocos de código dos [jupyter notebooks,](https://jupyter.org/) estão disponíveis sob a [licença GNU-GPL](LICENSE-CODE). Mais informações sobre essa licença em [GNU GENERAL PUBLIC LICENSE Version 3](https://www.gnu.org/licenses/gpl.html).
